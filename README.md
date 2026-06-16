# PCG Heart Murmur Detection

This repository contains a reproducible baseline project for heart sound classification using phonocardiogram (PCG) recordings from the CirCor heart sound dataset.

**Important clarification:**  
The current reported baseline results are from the `FusionResNetClassifier` model implemented in:

```text
src/models/resnet_fusion.py
```

This is **not** the later full dual-branch metadata multitask model.  
The current baseline is a ResNet-fusion model with:

- a spectrogram ResNet18 branch
- a handcrafted-feature MLP branch
- a fusion classifier

The later idea of using `PCG branch + TF branch + metadata branch + multitask heads` is a future extension and should not be confused with the currently reported baseline results.

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

## 3. Current Baseline Model

The current baseline model is:

```text
FusionResNetClassifier
```

implemented in:

```text
src/models/resnet_fusion.py
```

The model code has two internal branches:

```text
1. Spectrogram branch:
   log-Mel spectrogram → ResNet18 → deep embedding

2. Handcrafted branch:
   handcrafted PCG features → MLP → handcrafted embedding

3. Fusion classifier:
   concatenate embeddings → MLP classifier → logits
```

This model should be described as:

```text
ResNet18 + handcrafted feature fusion baseline
```

It should **not** be described as the final dual-branch metadata multitask model.

---

## 4. Model Definition

The current baseline model is equivalent to the following structure:

```python
import torch
import torch.nn as nn
import torchvision.models as models


class SpectrogramResNet18(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        return self.backbone(x)


class FusionResNetClassifier(nn.Module):
    def __init__(
        self,
        handcrafted_dim: int,
        num_classes: int = 2,
        deep_dim: int = 256,
        hand_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.deep_branch = SpectrogramResNet18(embedding_dim=deep_dim)

        self.hand_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, hand_dim),
            nn.BatchNorm1d(hand_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hand_dim, hand_dim),
            nn.BatchNorm1d(hand_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_dim = deep_dim + hand_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, logmel, handcrafted):
        deep_feat = self.deep_branch(logmel)
        hand_feat = self.hand_branch(handcrafted)

        fused = torch.cat([deep_feat, hand_feat], dim=1)
        return self.classifier(fused)
```

---

## 5. Data

The repository keeps both raw data and processed data.

Expected data structure:

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

### 5.1 Baseline Data

The baseline fusion model uses:

```text
log-Mel spectrogram input
handcrafted PCG feature input
```

Depending on the experiment configuration, these are generated from the available processed audio paths and segment metadata.

### 5.2 Processed Outcome Binary Data

The folder below contains the later processed data for clinical outcome experiments:

```text
data/processed_outcome_binary/
```

Main CSV:

```text
data/processed_outcome_binary/segments_outcome_binary_win5p0.csv
```

This file contains segment metadata, fold assignment, labels, and audio paths.

---

## 6. Tasks

The codebase can support several tasks, depending on configuration and label selection.

### 6.1 Murmur Binary

```text
Absent
Present
```

This excludes `Unknown` murmur labels.

### 6.2 Murmur 3-Class

```text
Absent
Present
Unknown
```

This includes the `Unknown` murmur class.

### 6.3 Clinical Outcome Binary

```text
Normal
Abnormal
```

The current OOP training entrypoint supports this through:

```bash
--task outcome_binary
```

---

## 7. Current Preprocessing Notes

The selected preprocessing configuration for the newer processed data is:

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

The repository may contain visualizations and processed folders related to dual-branch preprocessing. These are useful for inspection and future experiments, but the currently reported baseline results should still be attributed to:

```text
FusionResNetClassifier
```

not to the future metadata multitask model.

---

## 8. Installation

Create a virtual environment:

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

## 9. Verify Repository Data

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
]

for path in required_paths:
    p = Path(path)
    print(f"{path}: {'OK' if p.exists() else 'MISSING'}")

segments_path = Path("data/processed_outcome_binary/segments_outcome_binary_win5p0.csv")

if segments_path.exists():
    df = pd.read_csv(segments_path)

    print()
    print("Segments:", len(df))

    if "recording_id" in df.columns:
        print("Recordings:", df["recording_id"].nunique())

    if "patient_id" in df.columns:
        print("Patients:", df["patient_id"].nunique())

    if "outcome_label" in df.columns:
        print()
        print("Outcome labels by recording:")
        print(df[["recording_id", "outcome_label"]].drop_duplicates()["outcome_label"].value_counts())

    if "murmur_label" in df.columns:
        print()
        print("Murmur labels by recording:")
        print(df[["recording_id", "murmur_label"]].drop_duplicates()["murmur_label"].value_counts())

    if "fold" in df.columns:
        print()
        print("Fold counts:")
        print(df["fold"].value_counts().sort_index())
PY
```

Expected output should show all required paths as `OK`.

---

## 10. Smoke Test

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

The smoke test should print something similar to:

```text
Train segments: 128
Val segments: 64
Device: cuda
TRAIN BASELINE FUSION RESNET
Epoch 001 | loss ...
Saved best checkpoint ...
```

If the smoke test runs successfully, the OOP training entrypoint is working.

---

## 11. Train One Full Fold

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

On a shared server, use:

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

## 12. Train 5-Fold Baseline

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

## 13. Training Outputs

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

## 14. Baseline Result Attribution

When reporting results, use the following wording:

```text
The reported baseline results are obtained using FusionResNetClassifier,
which fuses a ResNet18 spectrogram embedding with handcrafted PCG features.
```

Do not say:

```text
The reported baseline results are from the full dual-branch metadata multitask model.
```

because that model has not been fully trained and reported yet.

---

## 15. Evaluation

The baseline trainer evaluates predictions at the patient level.

For binary classification, the metrics include:

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

## 16. Future Extension: Full Multitask Model

The next planned model is not the same as the current baseline.

Future model idea:

```text
TF branch:
processed_tf_path
→ MFCC + Mel + Chroma or log-Mel sequence
→ CNN / Transformer / BiLSTM / Attention

PCG branch:
processed_pcg_path
→ handcrafted waveform and spectral features
→ MLP

Metadata branch:
age, sex, height, weight, pregnancy_status, location
→ metadata encoder

Fusion:
TF embedding + PCG embedding + metadata embedding
→ shared representation

Heads:
outcome head: Normal / Abnormal
murmur head: Absent / Present / Unknown
```

Suggested future files:

```text
src/models/dualbranch_metadata_multitask.py
src/training/train_multitask.py
```

This future model should be implemented separately and should not overwrite the current baseline.

---

## 17. Development Convention

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

## 18. Important Leakage Rule

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

## 19. Git LFS

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

## 20. Notes About GitHub Directory Truncation

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

## 21. Push Updated Repository

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

## 22. Quick Start

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

## 23. Current Status

The current repository contains:

```text
Raw data
Processed data
Configuration files
Visualization figures
FusionResNet baseline model
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

The currently tested training task is:

```text
outcome_binary
```

The newer selected preprocessing setup available in the repository is:

```text
sr4000_bp25_600_zscore
```

The future model should be implemented separately as a full PCG + TF + metadata multitask model.
