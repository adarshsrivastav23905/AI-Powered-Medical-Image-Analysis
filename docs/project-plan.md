# Build and evidence plan

## Module map

- `src/data.py`: loads the dataset's existing directory splits, enforces label order, resizes images, and prefetches batches.
- `src/model.py`: augmentation, frozen pretrained MobileNetV2, pooling, dropout, binary output.
- `src/train.py`: training and validation, best checkpoint, early stopping, CSV log.
- `src/evaluate.py`: held-out test metrics, report, confusion matrix.
- `src/predict.py`: one-image inference with explicit educational-use notice.

## Reproducibility notes to fill in after a run

- Dataset source/version and terms reviewed:
- Train/validation/test counts by class:
- Python/TensorFlow versions:
- Machine and accelerator:
- Command and epoch count:
- Test threshold and metrics:
- Error examples and observed limitations:

Never invent these values. Record them from the actual run.
