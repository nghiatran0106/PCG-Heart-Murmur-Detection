# PCG Heart Murmur Detection

This repository contains a reproducible baseline pipeline for heart sound classification using phonocardiogram (PCG) recordings from the CirCor heart sound dataset.

The repository is organized for clean, object-oriented development. The current runnable baseline is a dual-branch fusion model that combines:

- A deep ResNet18 branch using log-Mel spectrograms
- A handcrafted PCG feature branch
- A fusion classifier for binary classification

The current training entrypoint is:

```text
src/training/train_baseline.py
```

---

## 1. Repository

```text
https://github.com/nghiatran0106/PCG-Heart-Murmur-Detection
```

Clone the repository:

```bash
git clone https://github.com/nghiatran0106/PCG-Heart-Murmur-Detection.git
cd PCG-Heart-Murmur-Detection
```

Pull large files tracked by Git LFS:

```bash
git lfs pull
```

---

## 2. Project Structure

```text
PCG-Heart-Murmur-Detection/
├── configs/
│   ├── baseline_fusion.yaml
│   ├── debug.yaml
│   └── full_resnet_fusion.yaml
│
├── configs_preproc/
│   ├── dual_branch_bp25_600_spectral.yaml
│   └── preproc_ablation.yaml
│
├── configs_safe/
│   └── baseline_fusion_*.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── processed_outcome_binary/
│
├── figures/
│   ├── dual_branch_after_by_class/
│   ├── dual_branch_processed_vis/
│   └── outcome_preprocess_visualization/
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
│
├── run_baseline_5fold.sh
├── requirements.txt
└── README.md
```

---

## 3. Data

The project uses the CirCor DigiScope heart sound dataset.

The expected data structure is:

```text
data/
├── raw/
│   └── circor-heart-sound/
│       └── training_data/
│
├── processed/
│   ├── metadata_clean.csv
│   └── segments.csv
│
└── processed_outcome_binary/
    ├── metadata_outcome_binary.csv
    ├── segments_outcome_binary_win5p0.csv
    ├── pcg_wav/
    │   ├── normal/
    │   └── abnormal/
    └── tf_wav/
        ├── normal/
        └── abnormal/
```

The current runnable baseline trains from the processed final dataset:

```text
data/processed_outcome_binary/
```

The main segment file is:

```text
data/processed_outcome_binary/segments_outcome_binary_win5p0.csv
```

This file contains segment-level metadata, fold assignment, labels, and paths to the processed PCG and TF branch audio files.

---

## 4. Current Preprocessing Setup

The selected final preprocessing configuration is:

```text
sr4000_bp25_600_zscore
```

Details:

```text
Sampling rate: 4000 Hz
Band-pass filter: 25–600 Hz
Normalization: z-score
Window length: 5 seconds
```

The final data uses a dual-branch design.

### 4.1 PCG Branch

```text
Raw PCG
→ resample to 4000 Hz
→ Butterworth band-pass 25–600 Hz
→ z-score normalization
→ handcrafted feature extraction
```

Stored in:

```text
data/processed_outcome_binary/pcg_wav/
```

### 4.2 TF Branch

```text
Raw PCG
→ resample to 4000 Hz
→ Butterworth band-pass 25–600 Hz
→ spectral gating
→ z-score normalization
→ log-Mel spectrogram
→ ResNet18 branch
```

Stored in:

```text
data/processed_outcome_binary/tf_wav/
```

---

## 5. Current Baseline Task

The current runnable baseline is configured for clinical outcome binary classification:

```text
Normal
Abnormal
```

The training command uses:

```bash
--task outcome_binary
```

The trainer also supports:

```text
outcome_binary
murmur_binary
murmur_3class
```

However, the currently tested baseline command is for:

```text
outcome_binary
```

---

## 6. Model Architecture

The baseline model is implemented in:

```text
src/models/resnet_fusion.py
```

The main class is:

```python
FusionResNetClassifier
```

The model has two branches.

### 6.1 Deep Branch

```text
Input: log-Mel spectrogram
Backbone: ResNet18
Input channel: 1
Output embedding dimension: 256
```

### 6.2 Handcrafted Branch

```text
Input: handcrafted PCG features
Architecture: MLP + BatchNorm + ReLU + Dropout
Output embedding dimension: 128
```

### 6.3 Fusion Classifier

```text
ResNet embedding + handcrafted embedding
→ concatenation
→ LayerNorm
→ MLP classifier
→ logits
```

Simplified forward pass:

```python
deep_feat = self.deep_branch(logmel)
hand_feat = self.hand_branch(handcrafted)
fused = torch.cat([deep_feat, hand_feat], dim=1)
logits = self.classifier(fused)
```

---

## 7. Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv_pcg
source .venv_pcg/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Set Python path:

```bash
export PYTHONPATH=$(pwd)
```

Check PyTorch and CUDA:

```bash
python - << 'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 8. Verify Repository Data

Before training, verify that all required files exist:

```bash
python - << 'PY'
from pathlib import Path
import pandas as pd

required_paths = [
    "configs/baseline_fusion.yaml",
    "src/training/train_baseline.py",
    "src/models/resnet_fusion.py",
    "data/processed_outcome_binary/segments_outcome_binary_win5p0.csv",
    "data/processed_outcome_binary/pcg_wav",
    "data/processed_outcome_binary/tf_wav",
]

for path in required_paths:
    p = Path(path)
    print(f"{path}: {'OK' if p.exists() else 'MISSING'}")

segments_path = Path("data/processed_outcome_binary/segments_outcome_binary_win5p0.csv")

if segments_path.exists():
    df = pd.read_csv(segments_path)
    print()
    print("Segments:", len(df))
    print("Recordings:", df["recording_id"].nunique() if "recording_id" in df.columns else "N/A")
    print("Patients:", df["patient_id"].nunique() if "patient_id" in df.columns else "N/A")

    if "outcome_label" in df.columns:
        print()
        print("Outcome labels by recording:")
        print(df[["recording_id", "outcome_label"]].drop_duplicates()["outcome_label"].value_counts())

    if "fold" in df.columns:
        print()
        print("Fold counts:")
        print(df["fold"].value_counts().sort_index())
PY
```

Expected output should show all required paths as `OK`.

---

## 9. Smoke Test

Before running a full experiment, run a small smoke test:

```bash
python -m src.training.train_baseline \
  --config configs/baseline_fusion.yaml \
  --fold 0 \
  --task outcome_binary \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --no-amp \
  --max-train-segments 128 \
  --max-val-segments 64
```

This should print information similar to:

```text
Train segments: 128
Val segments: 64
Device: cuda
TRAIN BASELINE FUSION RESNET
Epoch 001 | loss ...
Saved best checkpoint ...
```

If the smoke test runs successfully, the training pipeline is ready.

---

## 10. Train One Full Fold

Train fold 0:

```bash
python -m src.training.train_baseline \
  --config configs/baseline_fusion.yaml \
  --fold 0 \
  --task outcome_binary \
  --epochs 40 \
  --batch-size 32 \
  --num-workers 2 \
  --amp
```

If the server is shared, use a safer command:

```bash
nice -n 15 ionice -c2 -n7 python -m src.training.train_baseline \
  --config configs/baseline_fusion.yaml \
  --fold 0 \
  --task outcome_binary \
  --epochs 40 \
  --batch-size 32 \
  --num-workers 2 \
  --amp
```

---

## 11. Train 5-Fold Baseline

Run all 5 folds:

```bash
chmod +x run_baseline_5fold.sh
./run_baseline_5fold.sh
```

If running through SSH and you want the process to continue after disconnecting:

```bash
nohup ./run_baseline_5fold.sh > outputs/logs/baseline_5fold_nohup.log 2>&1 &
```

Monitor logs:

```bash
tail -f outputs/logs/baseline_5fold_nohup.log
```

---

## 12. Training Outputs

The trainer saves outputs into:

```text
outputs/
├── checkpoints/
├── logs/
├── predictions/
└── reports/
```

Expected files include:

```text
outputs/checkpoints/baseline_fusion_fold0_best.pt
outputs/logs/baseline_fusion_fold0_history.json
outputs/predictions/baseline_fusion_fold0_patient_predictions.csv
outputs/reports/baseline_fusion_fold0_metrics.json
```

For each fold, the trainer saves:

```text
Best checkpoint
Training history
Patient-level predictions
Patient-level metrics
```

---

## 13. Evaluation

The baseline trainer evaluates predictions at the patient level.

The current binary metrics include:

```text
Accuracy
Precision
Recall / Sensitivity
Specificity
F1-score
Balanced Accuracy
Matthews Correlation Coefficient
Cohen Kappa
AUROC
Average Precision
```

Evaluation utilities are stored in:

```text
src/evaluation/
├── aggregate.py
└── metrics.py
```

---

## 14. Development Convention

All new code should follow an object-oriented and modular design.

Recommended module responsibilities:

```text
src/data/
→ dataset classes and data loading logic

src/features/
→ feature extraction classes

src/preprocessing/
→ signal preprocessing classes

src/models/
→ PyTorch model definitions

src/training/
→ trainer classes and experiment entrypoints

src/evaluation/
→ metrics and patient-level aggregation

src/utils/
→ config loading, seed control, shared utilities
```

Avoid adding large procedural scripts to the project root.

Recommended class names for future extensions:

```text
BaselineTrainer
BaselineExperimentRunner
DualBranchPCGDataset
LogMelExtractor
HandcraftedPCGFeatureExtractor
PatientLevelAggregator
MetricComputer
MultitaskTrainer
MetadataEncoder
TFEncoder
PCGFeatureEncoder
DualBranchMetadataMultitaskModel
```

---

## 15. Important Leakage Rule

Never use target labels as model input features.

Do not use these columns as metadata inputs:

```text
outcome_label
outcome_target
murmur_label
murmur_target
label
target
```

Allowed metadata input features for future metadata branch:

```text
age
sex
height
weight
pregnancy_status
location
```

Labels should only be used for loss computation and evaluation.

---

## 16. Git LFS

This repository contains many audio files and may contain model checkpoints.

Install and initialize Git LFS:

```bash
git lfs install
```

Recommended LFS tracking:

```bash
git lfs track "*.wav"
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.ckpt"
git lfs track "*.onnx"
git lfs track "*.npy"
git lfs track "*.npz"
git lfs track "*.png"
git lfs track "*.jpg"
git lfs track "*.jpeg"
git lfs track "*.pdf"
git lfs track "*.ipynb"
```

After changing LFS tracking rules:

```bash
git add .gitattributes
git commit -m "Update Git LFS tracking"
```

---

## 17. Notes About GitHub Directory Truncation

GitHub may show a message like:

```text
Sorry, we had to truncate this directory to 1,000 files.
```

This is not a training error and does not mean files are missing.

It only means the GitHub web interface does not display all files in a directory with many entries.

The files can still be cloned with:

```bash
git clone ...
git lfs pull
```

---

## 18. Push Updated Repository

After modifying code or README:

```bash
git status
git add README.md
git add src/training/train_baseline.py
git add run_baseline_5fold.sh
git add requirements.txt
git add .gitignore .gitattributes
git commit -m "Update README and baseline training instructions"
git push origin main
```

If you also want to push training outputs:

```bash
git add outputs
git commit -m "Add training outputs"
git push origin main
```

Before pushing, make sure the virtual environment is not staged:

```bash
git status --short | grep ".venv_pcg" || echo "OK: .venv_pcg is not staged"
```

If `.venv_pcg` appears, remove it from Git tracking:

```bash
git rm -r --cached .venv_pcg
```

---

## 19. Quick Start

For a new team member:

```bash
git clone https://github.com/nghiatran0106/PCG-Heart-Murmur-Detection.git
cd PCG-Heart-Murmur-Detection
git lfs pull

python3 -m venv .venv_pcg
source .venv_pcg/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$(pwd)

python -m src.training.train_baseline \
  --config configs/baseline_fusion.yaml \
  --fold 0 \
  --task outcome_binary \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --no-amp \
  --max-train-segments 128 \
  --max-val-segments 64
```

If the smoke test works, train the full 5-fold baseline:

```bash
chmod +x run_baseline_5fold.sh
./run_baseline_5fold.sh
```

---

## 20. Current Status

The current repository contains:

```text
Raw data
Processed final data
Configuration files
Visualization figures
Baseline FusionResNet model
OOP baseline training entrypoint
Evaluation utilities
Feature extraction utilities
Training outputs if pushed
```

The current baseline is:

```text
FusionResNetClassifier
```

The current main training entrypoint is:

```text
src/training/train_baseline.py
```

The current recommended training task is:

```text
outcome_binary
```

The selected final preprocessing setup is:

```text
sr4000_bp25_600_zscore
```
