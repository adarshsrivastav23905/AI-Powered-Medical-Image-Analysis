# Dataset setup

This project expects the publicly available Kaggle **Chest X-Ray Images (Pneumonia)** dataset, subject to the dataset host's terms. Obtain it yourself from Kaggle, read and follow the license/terms there, and extract it here so the directories look like:

```text
data/raw/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

The training script reads the split directories as supplied. Never combine the splits before evaluation. Dataset splits and labels have known limitations; inspect the dataset card/source and document the exact version you used. Raw images are ignored by Git.
