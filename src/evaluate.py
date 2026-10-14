"""Evaluate once on the held-out test split and save metrics/plots."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)
from src.data import load_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/raw/chest_xray"))
    p.add_argument("--model", type=Path, default=Path("models/pneumonia_mobilenet.keras"))
    args = p.parse_args()
    ds = load_split(args.data, "test")
    model = tf.keras.models.load_model(args.model)
    y_true = np.concatenate([y.numpy().reshape(-1) for _, y in ds]).astype(int)
    scores = model.predict(ds).reshape(-1)
    y_pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity_recall_pneumonia": float(tp / (tp + fn)) if tp + fn else None,
        "specificity_recall_normal": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "threshold": 0.5,
        "confusion_matrix_labels": ["NORMAL", "PNEUMONIA"],
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], zero_division=0))
    ConfusionMatrixDisplay(cm, display_labels=["NORMAL", "PNEUMONIA"]).plot(cmap="Blues", values_format="d")
    plt.title("Held-out test confusion matrix (educational demo)")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=160)
    plt.close()

    # Small qualitative panel: actual dataset label next to the model's thresholded output.
    images, labels = next(iter(ds))
    count = min(12, int(images.shape[0]))
    image_scores = model.predict(images[:count], verbose=0).reshape(-1)
    columns = 4
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            actual = int(labels[index].numpy().reshape(-1)[0])
            predicted = int(image_scores[index] >= 0.5)
            axis.imshow(images[index].numpy().astype("uint8"), cmap="gray")
            axis.set_title(f"Actual: {['NORMAL', 'PNEUMONIA'][actual]}\nPredicted: {['NORMAL', 'PNEUMONIA'][predicted]}")
    fig.tight_layout()
    fig.savefig("outputs/prediction_examples.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
