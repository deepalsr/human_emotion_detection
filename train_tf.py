"""Train the same emotion CNN with TensorFlow / Keras on data/train and data/test.

Usage:
    python train_tf.py
    python train_tf.py --data data --epochs 30 --batch-size 64 --out emotion_tf.keras

Install:  pip install tensorflow scikit-learn
(Apple Silicon: also  pip install tensorflow-metal  to use the Mac GPU.)

Outputs: <out> (the Keras model) and <out>.classes.json (class names in label order).
10% of train is held out for validation; the test set is evaluated once at the end.
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 48


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(0.25)(x)


def build_model(num_classes):
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)           # [0,255] -> [-1,1]
    # Augmentation: only active during training
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(10 / 360)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.2)(x)
    for f in (32, 64, 128, 256):
        x = conv_block(x, f)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder containing train/ and test/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="emotion_tf.keras")
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    print("Devices:", tf.config.list_physical_devices())

    common = dict(color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE),
                  label_mode="categorical", batch_size=args.batch_size)
    train_ds = keras.utils.image_dataset_from_directory(
        f"{args.data}/train", validation_split=args.val_frac, subset="training",
        seed=args.seed, shuffle=True, **common)
    val_ds = keras.utils.image_dataset_from_directory(
        f"{args.data}/train", validation_split=args.val_frac, subset="validation",
        seed=args.seed, shuffle=False, **common)
    test_ds = keras.utils.image_dataset_from_directory(
        f"{args.data}/test", shuffle=False, **common)
    classes = train_ds.class_names
    print("Classes:", classes)

    # Class weights from folder counts (full train set; close enough to the 90% used)
    counts = np.array([len(os.listdir(f"{args.data}/train/{c}")) for c in classes], dtype=float)
    cw = {i: float(counts.sum() / (len(classes) * n)) for i, n in enumerate(counts)}
    print("Class weights:", {classes[i]: round(w, 2) for i, w in cw.items()})

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(len(classes))
    steps = int(tf.data.experimental.cardinality(train_ds).numpy()) * args.epochs
    lr = keras.optimizers.schedules.CosineDecay(args.max_lr, decay_steps=steps)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.ModelCheckpoint(
        args.out, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=cw, callbacks=callbacks)

    # Final, one-time test evaluation with the best checkpoint
    best = keras.models.load_model(args.out)
    loss, acc = best.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {acc:.1%}")

    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
    y_pred = np.argmax(best.predict(test_ds, verbose=0), axis=1)
    try:
        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    except ImportError:
        print("(pip install scikit-learn for a per-class report)")

    with open(args.out + ".classes.json", "w") as f:
        json.dump(classes, f)
    print(f"Saved best model to {args.out} and class names to {args.out}.classes.json")


if __name__ == "__main__":
    main()