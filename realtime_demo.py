"""Real-time facial emotion detection from the webcam.

Usage:
    python realtime_demo.py                 # default camera, emotion_torch.pt next to this file
    python realtime_demo.py --camera 1      # try another camera index
    python realtime_demo.py --ckpt path/to/emotion_torch.pt

Controls: press  q  or  ESC  to quit.

Requirements:  pip install torch torchvision "opencv-python>=4.8,<5" pillow numpy
(OpenCV 5.x has no CascadeClassifier, so stay on 4.x.)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMG_SIZE = 48


# ---- Model: must match the notebook exactly so the checkpoint loads ----------
def conv_block(cin, cout):
    layers = []
    for i in range(2):
        layers += [
            nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        ]
    layers += [nn.MaxPool2d(2), nn.Dropout(0.25)]
    return nn.Sequential(*layers)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Same preprocessing as the notebook's test_transform
test_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# BGR colours for the box/label of each emotion (unknown names fall back to white)
COLORS = {
    "angry": (0, 0, 255), "disgust": (0, 140, 0), "fear": (180, 0, 180),
    "happy": (0, 200, 255), "neutral": (200, 200, 200), "sad": (255, 120, 0),
    "surprise": (0, 165, 255),
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = EmotionCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


@torch.no_grad()
def predict(model, gray_crop, device):
    tensor = test_transform(Image.fromarray(gray_crop)).unsqueeze(0).to(device)
    return torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()


def draw_bars(frame, classes, probs, origin=(10, 10)):
    """Small probability panel in the top-left corner."""
    x0, y0 = origin
    for i, (name, p) in enumerate(zip(classes, probs)):
        y = y0 + i * 22
        cv2.rectangle(frame, (x0, y), (x0 + 150, y + 16), (40, 40, 40), -1)
        cv2.rectangle(frame, (x0, y), (x0 + int(150 * p), y + 16),
                      COLORS.get(name, (255, 255, 255)), -1)
        cv2.putText(frame, f"{name} {p:.0%}", (x0 + 155, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Real-time emotion detection")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--ckpt", default=os.path.join(here, "emotion_torch.pt"))
    ap.add_argument("--camera", type=int, default=0, help="camera index (0 = default)")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="0-0.95; higher = steadier but slower to react (default 0.6)")
    args = ap.parse_args()

    if not hasattr(cv2, "CascadeClassifier"):
        sys.exit(f"OpenCV {cv2.__version__} has no CascadeClassifier. "
                 'Run: pip install "opencv-python>=4.8,<5"')
    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}\n"
                 "Train the notebook first (it saves emotion_torch.pt) and put it next to this script, "
                 "or pass --ckpt.")

    device = pick_device()
    model, classes = load_model(args.ckpt, device)
    print(f"Model loaded on {device}. Classes: {classes}")

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        sys.exit("Could not load the Haar face detector file.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera. On macOS, allow camera access for your "
                 "terminal/IDE in System Settings > Privacy & Security > Camera, "
                 "or try --camera 1.")

    smoothed = None          # exponentially smoothed probabilities of the main face
    last_t = time.time()
    fps = 0.0
    print("Running. Press q or ESC in the video window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break

            frame = cv2.flip(frame, 1)                      # mirror view
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            if len(faces) == 0:
                smoothed = None
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                main_face = max(faces, key=lambda f: f[2] * f[3])
                for (x, y, w, h) in faces:
                    probs = predict(model, gray[y:y + h, x:x + w], device)
                    if np.array_equal((x, y, w, h), main_face):
                        smoothed = probs if smoothed is None else \
                            args.smooth * smoothed + (1 - args.smooth) * probs
                        probs = smoothed
                    k = int(probs.argmax())
                    color = COLORS.get(classes[k], (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{classes[k]} {probs[k]:.0%}"
                    cv2.putText(frame, label, (x, max(y - 8, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                draw_bars(frame, classes, smoothed)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
            last_t = now
            cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 90, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Emotion detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()