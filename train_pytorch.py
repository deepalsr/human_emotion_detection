"""Train the PyTorch EmotionCNN on a folder dataset (data/train, data/test).

Usage:
    python train_pytorch.py
    python train_pytorch.py --data data --epochs 30 --batch-size 64 --out emotion_torch.pt

The checkpoint it writes ({"state_dict", "classes"}) loads directly in realtime_demo.py.

Differences from the notebook: 10% of train is held out as a validation set and
used to pick the best checkpoint, so the test set is only evaluated once at the end.
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMG_SIZE = 48


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
            conv_block(1, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device, collect=False):
    model.eval()
    correct, total = 0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        if collect:
            preds.append(p.cpu())
            labels.append(y.cpu())
    acc = correct / total
    if collect:
        return acc, torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder containing train/ and test/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on macOS)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="emotion_torch.pt")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print("Using device:", device)

    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Same files twice: one view with augmentation (train), one without (validation)
    full_train = datasets.ImageFolder(f"{args.data}/train", train_tf)
    full_val = datasets.ImageFolder(f"{args.data}/train", eval_tf)
    test_ds = datasets.ImageFolder(f"{args.data}/test", eval_tf)
    classes = full_train.classes
    print("Classes:", classes)

    # Stratified train/val split
    rng = np.random.RandomState(args.seed)
    targets = np.array(full_train.targets)
    train_idx, val_idx = [], []
    for c in range(len(classes)):
        idx = rng.permutation(np.where(targets == c)[0])
        n_val = int(len(idx) * args.val_frac)
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    pin = device == "cuda"
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, 128, shuffle=False, num_workers=args.workers, pin_memory=pin)
    test_dl = DataLoader(test_ds, 128, shuffle=False, num_workers=args.workers, pin_memory=pin)

    # Class weights from the training portion only
    counts = torch.bincount(torch.tensor(targets[train_idx]), minlength=len(classes)).float()
    weights = (counts.sum() / (len(classes) * counts)).to(device)
    print("Class weights:", dict(zip(classes, weights.cpu().numpy().round(2))))

    model = EmotionCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=len(train_dl))

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * y.size(0)

        val_acc = evaluate(model, val_dl, device)
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out)
            marker = "  <- saved"
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {running / len(train_ds):.3f} | "
              f"val {val_acc:.1%} | lr {optimizer.param_groups[0]['lr']:.5f} | "
              f"{time.time() - t0:.0f}s{marker}")

    # Final, one-time test evaluation with the best checkpoint
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_acc, preds, labels = evaluate(model, test_dl, device, collect=True)
    print(f"\nBest val accuracy: {best_val:.1%} | Test accuracy: {test_acc:.1%}")
    try:
        from sklearn.metrics import classification_report
        print(classification_report(labels, preds, target_names=classes, digits=3))
    except ImportError:
        print("(pip install scikit-learn for a per-class report)")
    print(f"Saved best model to {args.out}")


if __name__ == "__main__":
    main()