# AI-Powered Medical Image Analysis

> An educational chest X-ray image-classification project that uses TensorFlow and MobileNetV2 transfer learning to distinguish the dataset labels `NORMAL` and `PNEUMONIA`.

> **Research and education only.** This project is not a medical device. Its outputs must not be used to diagnose, triage, treat, or make decisions about a person. Model scores are not calibrated clinical probabilities.

---

## Author

**Adarsh Srivastav**

---

## Project Overview

This project demonstrates an end-to-end supervised image-classification workflow using labeled chest X-ray images. It loads the dataset's existing training, validation, and test folders, resizes images to 224×224 RGB, applies training-time augmentation, and trains a binary classification head on a frozen ImageNet-pretrained MobileNetV2 feature extractor.

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

## Project Explanation

### In simple terms

The project teaches a computer to distinguish between two labels in a public chest X-ray dataset: `NORMAL` and `PNEUMONIA`. It demonstrates how to prepare images, train an image classifier, check its results on a separate test split, and show an educational prediction in a local web page.

In real healthcare, image-analysis software is researched for possible support in hospitals, diagnostic labs, radiology centers, and health-tech products—for example, helping qualified teams review images or prioritize work. Any such use requires evidence and safeguards beyond a classroom model. This repository does not connect to a hospital, interpret scans for care, or demonstrate that AI reduces errors or improves patient outcomes. A clinician remains responsible for medical decisions.

### In technical terms

This is a binary supervised image-classification pipeline. A `tf.data` input pipeline decodes and resizes RGB images to 224×224, with random rotation and zoom applied during training only. An ImageNet-pretrained MobileNetV2 feature extractor is frozen; global average pooling, dropout, and a sigmoid output form the trained classification head. Validation AUC selects the saved checkpoint, and evaluation reports held-out test metrics at threshold 0.5.

### End-to-end workflow

1. **Collect:** Obtain the public dataset and review its current source, version, and terms.
2. **Inspect:** Preserve the supplied train, validation, and test folders; record class counts and provenance.
3. **Preprocess:** Decode images, resize to 224×224 RGB, and batch/prefetch them. Augment training images only.
4. **Extract features:** Pass images through the frozen MobileNetV2 feature extractor.
5. **Train:** Fit the classification head against training labels and use validation data for checkpoint selection and early stopping.
6. **Evaluate:** Assess the selected checkpoint once on the held-out test split and review class-wise metrics and errors.
7. **Predict:** Run a single authorized image through the same resize/preprocessing path.
8. **Visualize:** Review the local app output, training curves, confusion matrix, and test examples.

## Technology Choices

| Option | Example approach | Difficulty | Compute | Typical output |
|---|---|---|---|---|
| A — simplest | Python, Pillow, scikit-learn, hand-crafted features, small classifier | Beginner | CPU | Basic image classification baseline |
| B — selected | Python, TensorFlow/Keras, MobileNetV2 transfer learning, public chest X-ray dataset, Streamlit | Beginner to intermediate | CPU works; GPU may reduce training time | Train/evaluate CLI workflow and local image-demo web app |
| C — advanced | PyTorch, detection/segmentation architectures, larger multi-site datasets, experiment tracking and API deployment | Advanced | GPU typically needed | Research prototype for localization or multi-class analysis |

**Option B is used here** because it combines a manageable setup, established transfer-learning methods, and a presentable local interface. OpenCV, Flask, tumor detection, fracture detection, and MRI analysis are not part of the current implementation. The repository uses Pillow/TensorFlow image loading and Streamlit, and the model classifies this dataset's two labels only.

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

## Screenshots

The screenshots below document the app, project structure, README, tests, and CLI. Click any preview to open the full-size capture.

> **Image-sharing notice:** Screenshots `06`–`09` contain chest X-ray samples. They are omitted from this public gallery; include or publish them only after confirming that the dataset's current terms permit redistribution of the images shown.

<details>
<summary>Application overview and model results</summary>

<table>
<tr>
<td><a href="screenshots/01_thorax_ai_home.png"><img src="screenshots/01_thorax_ai_home.png" width="480" alt="Thorax AI home screen"></a><br><sub>01 · Branded home screen and research-use notice</sub></td>
<td><a href="screenshots/02_model_metrics.png"><img src="screenshots/02_model_metrics.png" width="480" alt="Held-out model metrics"></a><br><sub>02 · Held-out metrics and low-specificity warning</sub></td>
</tr>
<tr>
<td><a href="screenshots/03_confusion_matrix.png"><img src="screenshots/03_confusion_matrix.png" width="480" alt="Held-out confusion matrix"></a><br><sub>03 · Confusion matrix with labels and counts</sub></td>
<td><a href="screenshots/04_training_curves.png"><img src="screenshots/04_training_curves.png" width="480" alt="Training and validation curves"></a><br><sub>04 · Training and validation curves</sub></td>
</tr>
<tr>
<td><a href="screenshots/05_about_project.png"><img src="screenshots/05_about_project.png" width="480" alt="About the research demo"></a><br><sub>05 · Research-demo workflow and safety information</sub></td>
<td><a href="screenshots/10_project_overview.png"><img src="screenshots/10_project_overview.png" width="480" alt="Project overview in the app"></a><br><sub>10 · Project overview and educational-use notice</sub></td>
</tr>
</table>

</details>

<details>
<summary>Project and development evidence</summary>

<table>
<tr>
<td><a href="screenshots/11_project_structure.png"><img src="screenshots/11_project_structure.png" width="480" alt="Project files in VS Code"></a><br><sub>11 · Project structure in VS Code</sub></td>
<td><a href="screenshots/12_readme_overview.png"><img src="screenshots/12_readme_overview.png" width="480" alt="README project overview"></a><br><sub>12 · README overview and project scope</sub></td>
</tr>
<tr>
<td><a href="screenshots/13_readme_workflow.png"><img src="screenshots/13_readme_workflow.png" width="480" alt="README workflow and main modules"></a><br><sub>13 · Main modules and workflow</sub></td>
<td><a href="screenshots/14_tests_passed.png"><img src="screenshots/14_tests_passed.png" width="480" alt="Eight unit tests passing"></a><br><sub>14 · Test run: 8 tests passed</sub></td>
</tr>
<tr>
<td><a href="screenshots/15_cli_workflow.png"><img src="screenshots/15_cli_workflow.png" width="480" alt="Dataset inspection CLI help"></a><br><sub>15 · Dataset-inspection command help</sub></td>
<td></td>
</tr>
</table>

</details>

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
| `src/inspect_data.py` | Count valid/corrupt images by split and save a grayscale training preview |
| `main.py` | Dispatch `inspect`, `train`, `evaluate`, and `predict` commands |

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

The Streamlit app uses a custom dark navy/teal theme configured in `.streamlit/config.toml` and the project identity assets in `images/`.

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
│   ├── inspect_data.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/                     # Input pipeline and dataset-inspection tests
├── .gitignore
├── main.py                    # Command dispatcher
└── requirements.txt
```

Generated files and raw data may not exist until you run the workflow. The project includes both a command-line workflow and a local Streamlit web interface; it does not provide live clinical integration.

---

## Beginner-Friendly Implementation Plan

| Phase | What to do and why | Expected evidence / verification | Common mistake to avoid |
|---|---|---|---|
| 1. Setup | Create a Python 3.10/3.11 virtual environment and install `requirements.txt`. | TensorFlow imports and `python main.py --help` works. | Installing packages into a different Python than the one running the project. |
| 2. Dataset | Obtain the public dataset, review its terms, preserve its supplied splits, and record version and class counts. | `data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}` exists. | Combining folders or committing raw images. |
| 3. Input pipeline | Load labels in stable class order, resize to 224×224 RGB, batch and prefetch. | Inspect a batch shape and confirm labels map to `NORMAL`=0, `PNEUMONIA`=1. | Applying augmentation to validation or test data. |
| 4. Model | Build a frozen MobileNetV2 feature extractor with a small classification head. | Model summary and one batch produce a sigmoid score. | Treating the score as a clinical probability. |
| 5. Training | Train with the training split; use validation AUC for checkpoint selection and validation loss for early stopping. | Checkpoint, CSV log, and training curves are written locally. | Selecting a model based on test-set results. |
| 6. Evaluation | Load the selected checkpoint and assess the held-out test split at the documented threshold. | Metrics JSON, classification report, confusion matrix, and example plot. | Repeatedly tuning against the test split or reporting accuracy alone. |
| 7. Prediction | Run the CLI on one authorized image. | A dataset label and model score are displayed with an educational-use notice. | Using private/identifiable images without proper authorization. |
| 8. Visualization | Review training curves, confusion matrix, test examples, and the app. | Figures and local UI are readable and limitations remain visible. | Presenting selected images as proof of clinical validity. |
| 9. GitHub publishing | Commit code and documentation; keep data, local model, and generated results out of Git. | `git status` shows only intended project files; remote contains the commit. | Committing credentials, patient data, or large generated artifacts. |
| 10. Final review | Re-run the documented commands and complete the evidence record below. | Fresh clone setup instructions work; all reported results are reproducible. | Inventing metrics or omitting dataset/software provenance. |

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

### 3. Inspect the dataset

```bash
python main.py inspect
```

This checks all expected split/class folders, reports supported image counts and unreadable files, and writes a grayscale display preview to `outputs/dataset_preview.png`. The grayscale image is for visual inspection only; training and inference still use 224×224 RGB inputs. To change paths or preview size:

```bash
python main.py inspect --data data/raw/chest_xray --output outputs/dataset_preview.png --samples-per-class 4
```

### 4. Train

```bash
python main.py train --epochs 8
```

Optional arguments include `--data` for the dataset root, `--epochs` for the maximum number of epochs, and `--output` for the checkpoint path. For example:

```bash
python main.py train --data data/raw/chest_xray --epochs 8 --output models/pneumonia_mobilenet.keras
```

### 5. Evaluate

```bash
python main.py evaluate
```

Use `--data` to select the dataset root and `--model` to select a saved checkpoint:

```bash
python main.py evaluate --data data/raw/chest_xray --model models/pneumonia_mobilenet.keras
```

### 6. Predict one image

```bash
python main.py predict path/to/example_xray.jpeg
```

Pass `--model` to use a non-default checkpoint. The output is a dataset-label prediction and model score—not a diagnosis or calibrated clinical probability.

### 7. Launch the web interface

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser. The app provides an image-upload demo, a held-out evaluation snapshot, and a project overview. It loads the locally saved model when an image is submitted. Keep the app local and only upload images you are authorized to process.

Run `python main.py --help`, or append `--help` to a subcommand, for available CLI arguments. Equivalent module commands include `python -m src.inspect_data`, `python -m src.train`, `python -m src.evaluate`, and `python -m src.predict`.

---

## Generated Outputs

The default workflow writes:

- `models/pneumonia_mobilenet.keras` — trained model checkpoint
- `outputs/training_log.csv` — epoch-level training and validation log
- `outputs/training_curves.png` — training and validation accuracy/loss curves
- `outputs/dataset_preview.png` — grayscale display preview from the training split
- `outputs/metrics.json` — held-out test metrics and threshold
- `outputs/confusion_matrix.png` — held-out test confusion matrix
- `outputs/prediction_examples.png` — actual and predicted labels for a small test batch

These outputs are generated locally and ignored by Git. Training on CPU can take time; runtime and results vary with data, software, and hardware.

### How to explain the project visuals

- **Training curves:** The teal line is training performance and the amber line is validation performance, plotted by epoch. In the recorded run, training accuracy rises while validation accuracy stays at 75%; the validation split contains only 16 images (8 per class), so these validation points are coarse and unstable. The increasing gap is a reason for caution, not proof of generalization.
- **Confusion matrix:** Rows are dataset labels and columns are model labels. For the recorded test run, the cells are 92 correctly labeled `NORMAL`, 142 `NORMAL` images labeled `PNEUMONIA`, 6 `PNEUMONIA` images labeled `NORMAL`, and 384 correctly labeled `PNEUMONIA`. The large `NORMAL` false-positive count explains the low 39.3% specificity. The threshold is 0.5.
- **Metric cards:** Accuracy is 76.3%, pneumonia sensitivity is 98.5%, normal specificity is 39.3%, and ROC AUC is 0.950 for the recorded held-out split. These results describe this dataset/run only; the score is not calibrated and no clinical conclusion follows.
- **Prediction examples:** Each tile compares a dataset label with the model's thresholded label. Teal indicates a match and amber a mismatch; examples are illustrative, not a clinical assessment or representative sample.
- **Dataset preview:** Images are displayed in grayscale to aid visual inspection. The classifier itself consumes RGB inputs; the preview does not indicate a separate grayscale preprocessing pipeline.

The validation split is especially small and the test results show an uneven error profile. Present these limitations alongside any screenshot; do not describe the scores as diagnostic accuracy or claim that the system is ready to assist clinicians.

## Virtual Simulation Walkthrough

This project simulates a narrow machine-learning workflow using a public, labeled dataset: the folders stand in for an image source, the existing labels stand in for reference categories, and the local interface demonstrates how a model output could be presented for learning. It does **not** simulate a hospital, a doctor, a clinical workflow, or patient outcomes.

1. Download the dataset under its current terms and place it in the documented folder structure.
2. Verify train/validation/test class folders and record the dataset version, terms, and class counts.
3. Run `python main.py inspect` to review split counts, image readability, and the grayscale preview.
4. Train with `python main.py train --epochs 8`; inspect the training log and curves.
5. Evaluate the saved checkpoint with `python main.py evaluate`; review sensitivity, specificity, AUC, and the confusion matrix.
6. Run `python main.py predict path/to/authorized_xray.jpeg` or start the local Streamlit interface.
7. Capture non-identifying screenshots of setup, training output, metrics, plots, and the UI. Do not share raw X-rays unless the dataset terms explicitly permit redistribution.

The displayed class is the model's thresholded prediction, not a verified dataset label, doctor assistance, or a clinical probability. An image's dataset label and model prediction can differ.

---

## Testing

Run the unit tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover image loading, stable class labels, dataset inspection, corrupt-image reporting, and preview generation. To check syntax separately:

```bash
python -m compileall main.py src tests
```

Training and held-out evaluation still require the public dataset and compatible project dependencies. After setup, run the documented commands and record the environment and dataset provenance.

---

## Project Evidence and Reproducibility

The screenshot gallery above documents the app and development workflow. Screenshots containing sample X-rays are not displayed in the gallery unless their redistribution is permitted by the dataset terms. Keep raw X-rays and personal or patient-identifying data out of Git.

Record the following in [`docs/project-plan.md`](docs/project-plan.md) or an accompanying report after running the project:

- Dataset source/version, terms reviewed, and access date
- Train/validation/test class counts and split provenance
- Python and TensorFlow versions
- Machine and accelerator
- Command and training configuration
- Evaluation threshold and test metrics
- Error examples, limitations, and checks for leakage or duplicates

Never invent results or present selected examples as representative evidence of clinical performance.

### Proof-of-work checklist

- [ ] Dataset source, version, terms, access date, and split class counts are recorded.
- [ ] Python/TensorFlow versions, hardware, command, and training configuration are recorded.
- [ ] Training curves, held-out metrics, confusion matrix, and error examples come from a real run.
- [ ] Public screenshots contain no secrets or identifiable patient information, and any dataset imagery is shared only when its terms permit redistribution.
- [ ] README results agree with the saved artifacts and are explicitly described as dataset-specific.
- [ ] A fresh clone can follow the setup and run instructions.

The current screenshot files are in `screenshots/`. Keep optional shareable evidence there only after checking permissions and removing identifying information. Do not add source X-rays.

### Seven-day portfolio plan

| Day | Focus | Suggested commit | Proof |
|---|---|---|---|
| 1 | Environment and project structure | `docs: document environment setup` | Python/TensorFlow versions and CLI help |
| 2 | Dataset setup and provenance | `docs: record dataset setup guidance` | Folder layout and class counts (no images) |
| 3 | Input pipeline | `feat: build image data pipeline` | Batch shape and class mapping |
| 4 | Model architecture | `feat: add transfer-learning classifier` | Model summary and output shape |
| 5 | Training | `feat: train and checkpoint model` | Training log and learning curves |
| 6 | Evaluation and demo | `feat: add evaluation and local demo` | Metrics, confusion matrix, UI screenshot |
| 7 | Documentation and release polish | `docs: finalize project evidence and limitations` | Clean Git status and complete README |

This is a suggested learning sequence, not a claim that each stage was separately committed. Use small commits that describe real changes; do not rewrite Git history merely to imitate the schedule.

### GitHub publishing

Repository name: `AI-Powered-Medical-Image-Analysis`

Suggested description: `Educational chest X-ray classification with TensorFlow MobileNetV2, held-out evaluation, and a local Streamlit demo.`
Suggested topics: `computer-vision`, `medical-imaging`, `tensorflow`, `keras`, `mobilenetv2`, `streamlit`, `machine-learning`, `education`

For a fresh local copy of the GitHub repository:

```bash
git clone https://github.com/adarshsrivastav23905/AI-Powered-Medical-Image-Analysis.git
cd AI-Powered-Medical-Image-Analysis
git status
```

After making a focused change:

```bash
git add README.md docs/
git diff --cached --check
git commit -m "docs: describe project workflow"
git push origin main
```

Before publishing, review `git status` and the staged file list. Keep `.venv/`, credentials, raw data, trained weights, and generated outputs out of commits. Only upload screenshots or plots that are non-identifying and permitted by the dataset terms. Add a short demo video or screenshots to the README only when they are genuine and reproducible.

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

## Project Discussion: Interview Practice

**What did you build?**
A reproducible educational binary classifier for the `NORMAL` and `PNEUMONIA` labels in a public chest X-ray dataset. It includes data loading, transfer learning, held-out evaluation, CLI inference, and a local Streamlit demonstration.

**Why MobileNetV2?**
It is a compact pretrained image feature extractor, making transfer learning practical for a student project and CPU-based experimentation. This is a learning choice, not evidence that it is suitable for clinical deployment.

**How did you avoid evaluating on training data?**
The supplied train, validation, and test partitions are kept separate. Training uses the train split; validation selects the checkpoint and supports early stopping; the held-out test split is used for final evaluation.

**Why report sensitivity, specificity, and AUC in addition to accuracy?**
Accuracy alone can hide class-specific errors, especially when class counts differ. Sensitivity, specificity, confusion matrices, and ROC AUC provide complementary views, but none establishes clinical safety.

**What are the main limitations?**
The project uses one public dataset, has no external or prospective validation, and its score is uncalibrated. Dataset leakage, label quality, population shift, privacy, and clinical workflow requirements remain unresolved.

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
