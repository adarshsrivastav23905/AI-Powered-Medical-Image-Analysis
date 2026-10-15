# AI-Powered Medical Image Analysis

> An educational chest X-ray image-classification project that uses TensorFlow and MobileNetV2 transfer learning to distinguish the dataset labels `NORMAL` and `PNEUMONIA`.

> **Research and education only.** This project is not a medical device. Its outputs must not be used to diagnose, triage, treat, or make decisions about a person. Model scores are not calibrated clinical probabilities.

---

## Author

**Adarsh Srivastav**

---

## Project Overview

This project demonstrates an end-to-end supervised image-classification workflow using labeled chest X-ray images. It loads the dataset's existing training, validation, and test folders, resizes images to 224×224 RGB, applies training-time augmentation, and fine-tunes a binary classification head on a frozen ImageNet-pretrained MobileNetV2 feature extractor.

The command-line workflow supports model training, held-out test evaluation, and single-image inference. It saves training logs, plots, metrics, and a model checkpoint locally. The repository does not include the X-ray dataset or a trained model; users must obtain data under its applicable terms and train the model themselves.

> This is a classroom software demonstration—not a simulation of clinical practice, a substitute for a clinician, or evidence of real-world diagnostic performance.

---

## Objectives

- Demonstrate a reproducible image-classification pipeline.
- Keep the dataset's train, validation, and test partitions separate.
- Apply image augmentation during training, not evaluation or inference.
- Use transfer learning to build a lightweight binary image classifier.
- Select a checkpoint using validation AUC and evaluate on a held-out test split.
- Report sensitivity, specificity, accuracy, ROC AUC, and a confusion matrix.
- Provide a simple command-line workflow for training, evaluation, and image inference.
- Document data provenance, limitations, and responsible interpretation.

---

## Key Features

- Directory-based dataset loading with stable class order: `NORMAL` (0), `PNEUMONIA` (1)
- RGB image resizing to 224×224 and batched, prefetched input
- Training augmentation with random rotation and zoom
- Frozen ImageNet-pretrained MobileNetV2 feature extractor
- Global average pooling, dropout, and sigmoid classification output
- Validation-AUC checkpointing and early stopping
- Held-out test metrics, classification report, confusion matrix, and example predictions
- Single-image prediction with an explicit educational-use notice
- Generated model and result artifacts are excluded from Git by `.gitignore`

---

## System Architecture

```text
Dataset folders
  train / val / test
  NORMAL / PNEUMONIA
          │
          ▼
Decode and resize to 224×224 RGB
          │
          ├── Training split only: random rotation and zoom
          ▼
MobileNetV2 ImageNet feature extractor (frozen)
          │
          ▼
Global average pooling → dropout → sigmoid score
          │
          ├── Train on train / select checkpoint on val
          ├── Evaluate once on held-out test
          └── Predict one image
```

### Main Modules

| Module | Responsibility |
|---|---|
| `src/data.py` | Load a dataset split, preserve label order, resize and prefetch batches |
| `src/model.py` | Build and compile the MobileNetV2 transfer-learning model |
| `src/train.py` | Train the model, save the best checkpoint and training artifacts |
| `src/evaluate.py` | Evaluate a saved model on the test split and save metrics and plots |
| `src/predict.py` | Run inference on one image and print an educational-use notice |
| `main.py` | Dispatch `train`, `evaluate`, and `predict` commands |

---

## Working Principle

```text
Obtain the dataset and verify its terms and split provenance
                         ↓
Keep train, validation, and test partitions intact
                         ↓
Decode images, map labels, and resize to 224×224 RGB
                         ↓
Augment training images and train the binary classifier
                         ↓
Select the checkpoint using validation AUC
                         ↓
Evaluate once on the held-out test data
                         ↓
Report metrics and optionally predict one image
```

### Dataset Layout

Download the Kaggle **Chest X-Ray Images (Pneumonia)** dataset yourself and follow the dataset host's current terms. Arrange the files as described in [`data/README.md`](data/README.md):

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Record the dataset source, version, access date, terms, and class counts before training. Keep the supplied partitions intact; do not use the test split for training, checkpoint selection, or other development choices. Do not commit raw images or redistribute samples unless the applicable terms explicitly allow it.

---

## Model and Evaluation

The model applies random rotation and zoom augmentation, MobileNetV2 preprocessing, a frozen ImageNet-pretrained MobileNetV2 base, global average pooling, dropout, and a one-unit sigmoid output. It is compiled with Adam, binary cross-entropy, binary accuracy, and AUC.

Training uses the training split and validation split. The best checkpoint is selected by validation AUC, while early stopping monitors validation loss. Evaluation loads that saved checkpoint and evaluates the held-out test split at a fixed threshold of `0.5`.

The evaluator writes:

- Accuracy
- Pneumonia sensitivity/recall
- Normal specificity/recall
- ROC AUC when both classes are represented
- Confusion matrix and a classification report
- A confusion-matrix plot and a small panel of test-set examples

No benchmark result is included in this repository. Populate any results summary only after a real run, and record the dataset version, class counts, software versions, hardware, threshold, metrics, and failure examples. A result on one dataset is not evidence of clinical safety, calibration, or generalization.

---

## Software and Tools

- **Language:** Python 3.10 or 3.11 recommended
- **Deep learning:** TensorFlow / Keras
- **Model:** MobileNetV2 with ImageNet weights
- **Image processing:** Pillow and TensorFlow image utilities
- **Numerical computing:** NumPy
- **Metrics:** scikit-learn
- **Plots:** Matplotlib
- **Web interface:** Streamlit
- **Version control:** Git

The Streamlit app uses a native light theme configured in `.streamlit/config.toml` and the project identity assets in `images/`.

---

## Project Structure

```text
AI-Powered-Medical-Image-Analysis/
├── data/
│   └── README.md              # Dataset setup and split layout
├── docs/
│   └── project-plan.md        # Module map and reproducibility checklist
├── images/                    # Optional shareable project assets
│   ├── thorax_ai_logo.png     # Header and sidebar wordmark
│   └── thorax_ai_icon.png     # App icon
├── .streamlit/
│   └── config.toml            # Clinical-research color and typography theme
├── models/                    # Locally generated model; ignored by Git
├── notebooks/                 # Optional exploration notebooks
├── outputs/                   # Locally generated logs and plots
├── app.py                     # Streamlit web interface
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── main.py                    # Command dispatcher
└── requirements.txt
```

Generated files and raw data may not exist until you run the workflow. The project includes both a command-line workflow and a local Streamlit web interface; it does not provide live clinical integration.

---

## How to Run the Project

### 1. Install dependencies

Use 64-bit Python 3.10 or 3.11 in a virtual environment. TensorFlow package availability depends on the Python version and operating system; consult the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip) if installation fails.

**Windows PowerShell (CPU):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import tensorflow as tf; print(tf.__version__)'
```

### 2. Prepare the dataset

Download and arrange the data using the layout above and the instructions in [`data/README.md`](data/README.md). Training downloads MobileNetV2 ImageNet weights on first use, so internet access is required unless those weights are already cached.

### 3. Train

```bash
python main.py train --epochs 8
```

Optional arguments include `--data` for the dataset root, `--epochs` for the maximum number of epochs, and `--output` for the checkpoint path. For example:

```bash
python main.py train --data data/raw/chest_xray --epochs 8 --output models/pneumonia_mobilenet.keras
```

### 4. Evaluate

```bash
python main.py evaluate
```

Use `--data` to select the dataset root and `--model` to select a saved checkpoint:

```bash
python main.py evaluate --data data/raw/chest_xray --model models/pneumonia_mobilenet.keras
```

### 5. Predict one image

```bash
python main.py predict path/to/example_xray.jpeg
```

Pass `--model` to use a non-default checkpoint. The output is a dataset-label prediction and model score—not a diagnosis or calibrated clinical probability.

### 6. Launch the web interface

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser. The app provides an image-upload demo, a held-out evaluation snapshot, and a project overview. It loads the locally saved model when an image is submitted. Keep the app local and only upload images you are authorized to process.

Run `python main.py --help`, or append `--help` to a subcommand, for available CLI arguments. Equivalent module commands are `python -m src.train`, `python -m src.evaluate`, and `python -m src.predict`.

---

## Generated Outputs

The default workflow writes:

- `models/pneumonia_mobilenet.keras` — trained model checkpoint
- `outputs/training_log.csv` — epoch-level training and validation log
- `outputs/training_curves.png` — training and validation accuracy/loss curves
- `outputs/metrics.json` — held-out test metrics and threshold
- `outputs/confusion_matrix.png` — held-out test confusion matrix
- `outputs/prediction_examples.png` — actual and predicted labels for a small test batch

These outputs are generated locally and ignored by Git. Training on CPU can take time; runtime and results vary with data, software, and hardware.

---

## Testing

An automated unit-test suite is not currently included. To check Python syntax without loading the TensorFlow runtime, run from the repository root:

```bash
python -m compileall main.py src
```

Functional validation requires the project dependencies and a correctly arranged dataset. After setup, exercise training, evaluation, and inference with the commands above; verify the generated outputs and record the environment and dataset provenance.

---

## Project Evidence and Reproducibility

No screenshots, dataset, trained checkpoint, or run-specific metrics are included as evidence in this repository. After a genuine run, optionally add non-identifying screenshots or plots to `images/`, only when the dataset terms permit sharing. Keep raw X-rays and personal or patient-identifying data out of Git.

Record the following in [`docs/project-plan.md`](docs/project-plan.md) or an accompanying report after running the project:

- Dataset source/version, terms reviewed, and access date
- Train/validation/test class counts and split provenance
- Python and TensorFlow versions
- Machine and accelerator
- Command and training configuration
- Evaluation threshold and test metrics
- Error examples, limitations, and checks for leakage or duplicates

Never invent results or present selected examples as representative evidence of clinical performance.

---

## Limitations and Responsible Use

- This is a research/education example, not a medical device or diagnostic aid.
- The public dataset's labels, provenance, split construction, and population representativeness may be limited or uncertain.
- Patient overlap, duplicates, acquisition artifacts, and other leakage can inflate evaluation results.
- A test result on one dataset may not generalize to other hospitals, devices, populations, or workflows.
- The sigmoid output is not calibrated as a clinical probability.
- A threshold of `0.5` is a classroom default, not a clinically justified operating point.
- There is no external validation, prospective study, clinician workflow, regulatory review, or deployment security in this project.
- Do not use private or identifiable patient data without appropriate authorization and safeguards.

Any real clinical application would require substantial domain expertise, representative data, independent validation, calibration, subgroup and site-shift analysis, privacy and security controls, human-factors work, regulatory review, and prospective evaluation.

---

## Future Scope

- Add explicit dataset inspection for corrupt files, class counts, and duplicate risks.
- Improve reproducibility with recorded configuration, seeds, and environment details.
- Add additional class-wise metrics and structured error analysis.
- Investigate patient-level split provenance and external validation where appropriate data is available.
- Compare carefully selected models and calibration methods for research purposes.
- Add automated tests for data loading, preprocessing, and metric calculations.
- Create a dashboard only if it can communicate limitations and uncertainty responsibly.

---

## Skills Demonstrated

### Machine Learning and Computer Vision

- Supervised binary image classification
- Transfer learning with MobileNetV2
- Training-only data augmentation
- Validation-based checkpoint selection
- Held-out evaluation and error analysis

### Python Engineering

- Modular command-line project structure
- TensorFlow data input pipelines
- Model training, saving, loading, and inference
- Metrics and visualization
- Reproducibility and dataset handling

### Responsible Research Communication

- Dataset provenance and split integrity
- Sensitivity, specificity, and ROC AUC reporting
- Clear distinction between model scores and clinical probabilities
- Explicit limitations and non-clinical-use boundaries

---

## Documentation

- [`data/README.md`](data/README.md) — dataset setup and directory layout
- [`docs/project-plan.md`](docs/project-plan.md) — implementation map and reproducibility checklist

---

## Project Status

The repository contains a functional training, evaluation, and single-image inference workflow. The dataset and trained model are user-provided/generated locally. No run-specific performance results are asserted here, and the project has not been clinically validated.

---

## Conclusion

This project demonstrates how an image input pipeline, transfer-learning model, held-out evaluation, and command-line inference can be assembled into a small medical-imaging research exercise. It is intended to support learning and discussion—not clinical decisions.

```text
LABELED X-RAY FOLDERS
        ↓
INPUT PIPELINE
        ↓
MOBILENETV2 TRANSFER LEARNING
        ↓
VALIDATION-SELECTED CHECKPOINT
        ↓
HELD-OUT TEST EVALUATION
        ↓
EDUCATIONAL LABEL + MODEL SCORE
```
