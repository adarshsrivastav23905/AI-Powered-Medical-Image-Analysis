# Dataset setup

This project expects the publicly available Kaggle **Chest X-Ray Images (Pneumonia)** dataset, subject to the dataset host's terms. Obtain it yourself from Kaggle, read and follow the license/terms there, and extract it here so the directories look like:

```text
data/raw/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

The training script reads the split directories as supplied. Never combine the splits before evaluation. Dataset splits and labels have known limitations; inspect the dataset card/source and document the exact version you used. Raw images are ignored by Git.

From the repository root, inspect folder structure, count readable images by class, identify corrupt files, and save a grayscale preview from the training split:

```bash
python main.py inspect
```

The preview is for human inspection only. The model input pipeline remains RGB at 224×224 pixels.
