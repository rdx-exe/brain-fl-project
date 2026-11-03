# Technical Architecture: Brain FL Project

## Overview
This document describes the technical architecture, modules, execution flows, and data processing pipelines for the **Brain FL Project — Privacy-Preserving Federated Learning for Alzheimer's Disease Classification Using Brain MRI**.

## 1. Final Directory Structure

```text
brain-fl-project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── centralized/
│   ├── federated/
│   ├── differential_privacy/
│   └── mia/
│
├── data/
│   ├── raw/
│   │   └── archives/
│   ├── extracted/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── processed/
│   └── manifests/
│
├── preprocessing/
│   ├── __init__.py
│   ├── build_slice_manifest.py
│   ├── create_dataset_splits.py
│   ├── create_federated_partitions.py
│   ├── extract_mri_dataset.py
│   ├── pytorch_dataset.py
│   └── [audit/inspection utilities...]
│
├── models/
│   ├── __init__.py
│   └── resnet18_model.py
│
├── experiments/
│   ├── __init__.py
│   └── train_centralized.py
│
├── federated/
│   ├── __init__.py
│   ├── client.py
│   ├── server.py
│   ├── run_federated.py
│   ├── run_dp.py
│   ├── dp_client.py
│   ├── dp_server.py
│   ├── dp_config.py
│   ├── federated_config.py
│   ├── privacy_accountant.py
│   ├── compare_experiments.py
│   └── evaluate.py
│
├── privacy/
│   ├── __init__.py
│   ├── attack_dataset.py
│   └── membership_inference.py
│
├── evaluation/
│
├── results/
│   ├── centralized/
│   ├── federated/
│   ├── privacy/
│   └── quality_control/
│
├── reports/
│
├── notebooks/
│
├── scripts/
│
├── tests/
│   ├── test_membership_inference_protocol.py
│   └── test_mia_artifacts.py
│
└── docs/
    ├── REPOSITORY_AUDIT.md
    ├── CLEANUP_REPORT.md
    ├── architecture.md
    └── images/
        ├── README.md
        ├── project-banner.png
        ├── dataset-pipeline.png
        └── system-architecture.png
```

## 2. Module Responsibilities

### Preprocessing (`preprocessing/`)
- **`extract_mri_dataset.py`**: Extracts T1-weighted MRI volumes (Analyze/NIfTI format) from OASIS-1 tar archives.
- **`create_dataset_splits.py`**: Generates participant-level stratified train (70%), validation (15%), and test (15%) splits.
- **`build_slice_manifest.py`**: Extracts seven fixed axial slices per participant and builds a consolidated slice manifest.
- **`pytorch_dataset.py`**: PyTorch Dataset class providing slice normalization, resizing (224×224), and three-channel replication.
- **`create_federated_partitions.py`**: Allocates training participants across four simulated federated clients (IID round-robin or non-IID Dirichlet distribution).

### Models (`models/`)
- **`resnet18_model.py`**: OASIS-1 ResNet-18 classifier with binary classification head (CN vs. CDR-positive) and validation utilities.

### Centralized Training (`experiments/`)
- **`train_centralized.py`**: Trains the centralized ResNet-18 model using slice-level batches and participant-level validation aggregation.

### Federated Learning (`federated/`)
- **`client.py`**: Flower client implementation for participant-level federated training (FedAvg).
- **`server.py`**: FedAvg simulation server coordinating training rounds and centralized participant-level validation.
- **`run_federated.py`**: Command-line entry point for FedAvg simulation experiments.
- **`run_dp.py`**: Entry point for participant-level differentially private federated learning.
- **`dp_client.py`**: Flower client with Opacus differential privacy integration.
- **`dp_server.py`**: DP federated learning server with GroupNorm backbone support.
- **`dp_config.py`**: Differential privacy configuration parameters.
- **`privacy_accountant.py`**: Participant-level RDP privacy accountant.
- **`compare_experiments.py`**: Compares IID and non-IID federated training runs.
- **`evaluate.py`**: Evaluates federated models on the untouched test split.

### Privacy Analysis (`privacy/`)
- **`attack_dataset.py`**: Constructs participant-level attack datasets by querying target models and aggregating slice probabilities.
- **`membership_inference.py`**: Implements confidence-based membership inference attacks (MIA) against centralized, federated, and DP models.

## 3. Data Processing Pipeline

1. **Raw MRI Archives**: OASIS-1 cross-sectional dataset archives (`.tar.gz`).
2. **Extraction & Masking**: Extract masked GFC MRI volumes (`*_t88_masked_gfc.img/.hdr`).
3. **Participant-Level Splitting**: Stratified 70/15/15 split (Train: 164, Val: 35, Test: 36 participants).
4. **Slice Selection**: Seven fixed axial slices (`[60, 70, 80, 90, 100, 110, 120]`) extracted per participant.
5. **Normalization**: Per-slice intensity clipping at 1st and 99.5th percentiles of non-zero voxels, scaling to `[0, 1]`.
6. **Resizing & Formatting**: Bilinear interpolation to 224×224, grayscale replicated to 3 channels.
7. **Federated Partitioning**: Four-client assignment (IID round-robin or Dirichlet non-IID).

## 4. Model Training Flow

- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: AdamW (Learning rate: $1\times 10^{-4}$, Weight decay: $1\times 10^{-4}$).
- **Scheduler**: ReduceLROnPlateau (patience=2, factor=0.5).
- **Early Stopping**: Patience of 7 epochs on validation loss.
- **Validation**: Participant-level aggregation (mean probability of class 1 across seven slices).

## 5. Federated Communication Flow

1. **Client Initialization**: Four Flower clients initialized with local participant partitions.
2. **Round Initialization**: Server broadcasts global model parameters to all clients.
3. **Local Training**: Clients perform local training (1 epoch, batch size 8) on their slice datasets.
4. **Parameter Upload**: Clients return updated model parameters and training metrics to the server.
5. **Server Aggregation**: FedAvg aggregates client parameters weighted by participant count.
6. **Global Evaluation**: Server evaluates the aggregated global model on the validation split.
7. **Checkpointing**: Best global model saved based on participant-level validation loss.

## 6. Differential Privacy Integration

- **Privacy Unit**: Participant (all seven slices clipped and noised together).
- **Mechanism**: Opacus DP-SGD with RDP accountant.
- **Configuration**: Noise multiplier $\sigma = 1.0$, max gradient norm $C = 1.0$, target $\delta = 10^{-5}$.
- **Backbone**: GroupNorm-based ResNet-18 (BatchNorm replaced for compatibility with Opacus).

## 7. Membership Inference Attack Evaluation Flow

1. **Target Loading**: Load centralized, FedAvg, or DP model checkpoint.
2. **Feature Extraction**: Query model on train (members) and validation (non-members) split slices; aggregate to participant level.
3. **Attack Calibration/Evaluation Split**: 50/50 stratified split of participants into calibration (fit attack model) and evaluation (measure attack performance).
4. **Attack Model**: Class-balanced Logistic Regression on standardized confidence features.
5. **Metrics**: Accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, advantage, and low-FPR TPR.

## 8. Experiment Execution Dependencies

- **Data Availability**: Requires extracted MRI volumes and extraction reports.
- **Pre-computed Partitions**: Federated experiments require `data/partitions/federated/*.csv`.
- **Pre-trained Checkpoints**: MIA requires trained checkpoints in `results/`.
- **Hardware**: GPU recommended for training and differential privacy; CPU sufficient for evaluation and MIA.
