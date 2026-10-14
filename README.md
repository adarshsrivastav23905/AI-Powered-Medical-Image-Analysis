# AI-Powered Medical Image Analysis

An educational computer vision project that classifies chest X-ray images into the dataset labels **NORMAL** and **PNEUMONIA**. It uses transfer learning with MobileNetV2, a public dataset, and a held-out test evaluation.

> **Research/education only.** This software is not a medical device and must not be used to diagnose, triage, treat, or make decisions about a person. Model scores are not calibrated clinical probabilities. Use only with appropriate public or consented data, and follow the dataset's terms.

## A. What the project demonstrates

### In simple terms

Hospitals and diagnostic services handle many images. A computer vision model can learn visual patterns from labeled examples and return a label for review. A well-designed system can help researchers study image triage and workflow support, while a qualified clinician remains responsible for interpretation.

### Technical view

This is supervised binary image classification. The pipeline reads labeled image folders, resizes RGB images to 224×224, applies training-only augmentation, and passes them through an ImageNet-pretrained MobileNetV2 feature extractor and a sigmoid classification head. It reports accuracy, ROC AUC, sensitivity, specificity, and a confusion matrix on the supplied test split.

Typical workflow: data provenance and split review → decoding/resizing → augmentation → feature extraction → training → held-out evaluation → single-image inference → plots and documented limitations. Real clinical systems also require external validation, calibration, subgroup and site-shift analysis, privacy/security controls, human factors work, regulatory review, and prospective evaluation; this repository does not provide those.

## B. Stack options

| Option | Tools and model | Data | Difficulty / compute | Expected output |
|---|---|---|---|---|
| A: easiest | Python, Pillow, scikit-learn, small HOG + SVM baseline | Small curated public image set | Low; CPU | Baseline metrics and error analysis |
| B: selected | Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn; MobileNetV2 transfer learning | Public chest X-ray folders | Medium; CPU works, GPU shortens training | Reproducible classifier, metrics, confusion matrix, sample predictions |
| C: advanced | PyTorch, MONAI, DICOM tooling, segmentation/detection model, experiment tracking | Curated multi-site DICOM or research dataset | High; GPU and medical-imaging expertise recommended | Segmentation/detection study with richer validation |

Option B is a practical student portfolio project: it runs without hospital access, teaches the end-to-end workflow, and uses a lightweight pretrained network. A GPU is optional. Native Windows TensorFlow GPU support differs from Linux/WSL; see installation notes below.

## C–D. Selected approach and architecture

```text
Public labeled X-ray folders
       ↓
Dataset/split checks and image decoding
       ↓
Resize to 224×224 RGB → training augmentation
       ↓
MobileNetV2 ImageNet feature extractor (frozen)
       ↓
Pooling → dropout → sigmoid output
       ↓
Train on train / choose checkpoint on val / evaluate once on test
       ↓
NORMAL or PNEUMONIA dataset-label score + metrics/plots
```

`src/data.py` handles folder loading and stable class mapping. `src/model.py` builds the network. `src/train.py` trains and saves the best validation-AUC checkpoint. `src/evaluate.py` evaluates the held-out test split. `src/predict.py` runs one image. The sigmoid score is a model score, not a calibrated probability or diagnosis.

## E. Repository structure

```text
AI-Powered-Medical-Image-Analysis/
├── data/                 # Dataset instructions; raw data stays local
├── docs/                 # Project plan and proof checklist
├── images/               # Optional README screenshots/demo images
├── models/               # Locally generated model (ignored by Git)
├── notebooks/            # Optional exploration notebooks
├── outputs/              # Metrics, training log, confusion matrix
├── src/                  # Data, model, training, evaluation, inference
├── main.py               # Convenience command dispatcher
├── requirements.txt
└── README.md
```

## F. Installation

Use 64-bit Python 3.10 or 3.11 in a virtual environment. Install packages from `requirements.txt`. TensorFlow's current pip installation instructions support CPU use on native Windows; GPU support for newer TensorFlow on Windows is through WSL2, while native-Windows CUDA support ended after TensorFlow 2.10. Refer to the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip) for current platform and Python compatibility.

### Windows PowerShell (CPU)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

If script activation is restricted, use `\.venv\Scripts\python.exe` directly for the pip and run commands.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import tensorflow as tf; print(tf.__version__)'
```

## Dataset

Download the Kaggle **Chest X-Ray Images (Pneumonia)** dataset yourself and comply with its current host and dataset terms. Do not commit the images. Arrange it as described in [`data/README.md`](data/README.md). The code expects `train`, `val`, and `test` directories, each containing `NORMAL` and `PNEUMONIA` class folders. Confirm the source, version, split provenance, and allowed use before training. Do not re-mix or tune on the test split.

This dataset is useful for a classroom simulation, not evidence of clinical performance. Dataset quality, split construction, duplicate/patient leakage, source artifacts, and population representativeness can affect results. Record what you verify and what remains unknown.

## G. Run the project

From the repository root with the environment active:

```bash
python main.py train --epochs 8
python main.py evaluate
python main.py predict path/to/example_xray.jpeg
```

The equivalent module commands are `python -m src.train`, `python -m src.evaluate`, and `python -m src.predict path/to/image.jpeg`. Override data/model paths with `--data` and `--model`. `--help` shows command arguments. Training downloads pretrained MobileNetV2 weights on first run, so internet access is needed then. CPU training may take a while. Output file names are examples; actual metrics depend on the data, software, and hardware, so no accuracy is promised.

Generated artifacts:

- `models/pneumonia_mobilenet.keras`: best validation-AUC checkpoint
- `outputs/training_log.csv`: epoch-level loss/metrics
- `outputs/training_curves.png`: training and validation accuracy/loss curves
- `outputs/metrics.json`: test accuracy, sensitivity, specificity, AUC, threshold
- `outputs/confusion_matrix.png`: held-out test confusion matrix
- `outputs/prediction_examples.png`: actual and predicted labels for a small test batch

The default threshold is 0.5 for a classroom baseline only. Do not present a threshold as clinically appropriate. For robust reporting include class counts, precision/recall, false negatives/positives, data split provenance, and limitations.

## H. Virtual simulation workflow

1. Obtain the public dataset and record its source, access date, terms, and folder counts.
2. Keep the supplied train/validation/test partitions intact; inspect for obvious corruption and duplicates where feasible.
3. Train only on the training split. Augmentation runs during training only.
4. Use validation metrics for checkpoint selection and development choices.
5. Evaluate once on the held-out test partition and preserve the confusion matrix and class-wise metrics.
6. Run one or more representative sample images and label outputs clearly as model predictions.
7. Discuss false negatives, false positives, imbalance, data leakage risk, bias, and why an actual clinician must not rely on this demo.

This simulates a narrow software workflow with labeled images. It does not simulate a hospital, a doctor, a patient pathway, or real-world clinical validation.

## I. Phased implementation plan

| Phase | Work / why | Expected result and verification | Common mistake |
|---|---|---|---|
| 1 Setup | Create venv and install dependencies for isolated reproducibility | TensorFlow imports and version prints | Mixing global and venv packages |
| 2 Data | Download terms-compliant dataset and place splits | Expected class folders/counts present | Committing images or mixing splits |
| 3 Input prep | Decode, resize, map labels; augment training only | Batch shapes `(batch,224,224,3)` | Augmenting validation/test |
| 4 Model | Load pretrained MobileNetV2 and add binary head | Model summary has one sigmoid output | Treating score as diagnosis |
| 5 Training | Fit train split; select best validation AUC | Keras epoch log and saved checkpoint | Training on test data |
| 6 Evaluation | Compute confusion matrix and class metrics on test | JSON and PNG appear in outputs | Reporting accuracy alone |
| 7 Prediction | Run an individual image | Dataset-label result and score | Using private patient data |
| 8 Visualization | Inspect metrics and representative errors | Readable plots and discussion | Cherry-picking only successes |
| 9 GitHub | Commit source/docs, exclude data and large model | Clean, reproducible repository | Uploading restricted data/secrets |
| 10 Final | Document limitations and reproducibility details | Portfolio-ready research demo | Claiming clinical readiness |

## J. GitHub publishing

Suggested repo name: `ai-medical-image-analysis`. Description: `Educational chest X-ray image classification demo using TensorFlow and MobileNetV2 transfer learning.` Topics: `computer-vision`, `medical-imaging`, `tensorflow`, `deep-learning`, `chest-xray`, `education`.

Create an empty GitHub repository, then from this directory:

```bash
git init
git add README.md .gitignore requirements.txt main.py src data docs images notebooks outputs
git commit -m "Create medical image analysis project scaffold"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-medical-image-analysis.git
git push -u origin main
```

Use small commits that reflect completed work. Keep raw data, trained weights, personal information, and credentials out of Git. Put compact, non-identifying plots in `images/` only when the dataset terms allow redistribution; otherwise make plots locally and link to the generation instructions. A short screen recording should show commands and generated metrics, not imply clinical use.

## K. Results and responsible interpretation

No result is included in advance. Populate this section after a real run with software versions, dataset version/split, class counts, hardware, threshold, test metrics, and observed failure modes. Do not fabricate performance or compare to clinicians. Explain that a retrospective benchmark result on one dataset does not establish safety or generalization to other hospitals, equipment, ages, or populations.

## L. Seven-day proof plan

| Day | Commit idea | Proof to capture |
|---|---|---|
| 1 Setup | `Add project scaffold and environment instructions` | Python version, dependency install |
| 2 Dataset | `Document public dataset setup and provenance` | Folder tree and class counts, no images if terms disallow |
| 3 Preprocessing | `Add deterministic image loading pipeline` | Example resized batch dimensions |
| 4 Model | `Add MobileNetV2 transfer-learning model` | Model summary and architecture diagram |
| 5 Training | `Add training and checkpoint workflow` | Training log plot, training command |
| 6 Evaluation | `Add held-out metrics and confusion matrix` | Metrics JSON and confusion matrix, include errors |
| 7 Publish | `Document results and publish repository` | Clean GitHub page, README rendered, final demo |

## M. Screenshots and demo assets

Use descriptive names such as `dataset_folder_layout.png`, `training_curves.png`, `test_confusion_matrix.png`, `sample_prediction_demo.png`, and `github_repository_overview.png`. Store shareable README assets under `images/`; store generated metrics/plots under `outputs/`. Check dataset terms before redistributing even a single sample image. Do not include identifiable patient data. Add screenshots to this README only after generating them, and caption them with split/source and context.

## Interview discussion prompts

- Why use transfer learning for a small dataset?
- Why are sensitivity and specificity relevant alongside accuracy?
- What does a false negative mean in this dataset-label task?
- How could patient-level leakage inflate test metrics?
- What additional validation would be required before any clinical use?

## License and data

Choose and add a software license before inviting reuse. A software license does not grant rights to the dataset. Follow the image dataset's separate terms and cite its source in your own report.
