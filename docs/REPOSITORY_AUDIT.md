# Repository Audit Report: Brain FL Project

## Overview
This document provides a comprehensive audit of the Brain FL Project repository, which implements privacy-preserving federated learning for Alzheimer's Disease classification using brain MRI data from the OASIS-1 dataset.

## 1. Repository Structure Analysis

### Core Modules

**Models (`models/`):**
- `resnet18_model.py`: Custom ResNet-18 implementation for OASIS-1 classification
- `model_utils.py`: Model factory and validation utilities (empty)
- `__init__.py`: Empty

**Preprocessing (`preprocessing/`):**
- `build_slice_manifest.py`: Constructs participant-level slice manifest (7 fixed axial slices)
- `create_dataset_splits.py`: Creates train/validation/test splits from OASIS-1
- `create_federated_partitions.py`: Generates IID and non-IID federated client partitions
- `extract_mri_dataset.py`: Extracts MRI volumes from archives
- `full_dataset_audit.py`: Comprehensive dataset quality control
- `metadata_loader.py`: Loads and validates metadata
- `pytorch_dataset.py`: PyTorch Dataset class for slice extraction and preprocessing
- `validate_dataset_manifest.py`: Validates dataset manifests
- `validate_mri_volumes.py`: Validates MRI volumes
- `visualize_training_volumes.py`: Visualizes training volumes
- `analyze_inspector.py`: Analyzes inspection data
- `analyze_intensity_distribution.py`: Analyzes intensity distributions
- `analyze_slice_coverage.py`: Analyzes slice coverage
- `verify_brain_masks.py`: Verifies brain masks
- `dataset_quality_control.py`: Main dataset quality control pipeline
- `inspect_mri_sample.py`, `archive_inspector.py`: Sample inspection utilities
- `__init__.py`: Empty

**Experiments (`experiments/`):**
- `train_centralized.py`: Centralized ResNet-18 training with participant-level validation
- `__init__.py`: Empty

**Federated Learning (`federated/`):**
- `server.py`: FedAvg simulation server with centralized validation
- `client.py`: Flower client implementation for participant-level partitions
- `run_federated.py`: Entry point for federated experiments
- `run_dp.py`: Entry point for differential privacy experiments
- `dp_client.py`: DP-enabled Flower client with Opacus
- `dp_server.py`: DP federated learning server
- `dp_config.py`: Differential privacy configuration
- `privacy_accountant.py`: Participant-level privacy accounting
- `federated_config.py`: Federated learning configuration
- `compare_experiments.py`: IID vs non-IID experiment comparison
- `evaluate.py`: Federated model evaluation on test split
- `__init__.py`: Documentation only

**Privacy (`privacy/`):**
- `membership_inference.py`: Participant-level MIA implementation against centralized, federated, and DP models
- `attack_dataset.py`: Attack dataset construction for MIA
- `__init__.py`: Documentation only

**Evaluation (`evaluation/`):**
- Directory exists but appears empty

**Scripts (`scripts/`):**
- `inspect_metadata.py`: Metadata inspection utility
- `inspect_analyze_pairs.py`: Pair analysis utility
- `inspect_archives.py`: Archive inspection utility

**Tests (`tests/`):**
- `test_membership_inference_protocol.py`: Protocol tests for MIA
- `test_mia_artifacts.py`: Artifact tests for MIA

**Other:**
- `backups/`: Experiment backups
- `configs/`: Configuration files (config.yaml empty)
- `models/`, `preprocessing/`, `experiments/`, `federated/`, `privacy/`: Core modules
- `results/`: All experiment outputs organized by experiment type
- `data/`: Raw, extracted, processed datasets and partitions
- `attacks/`: Attack-related files (empty directory)

## 2. Main Execution Flows

### Centralized Training
1. `preprocessing/` → Dataset preparation → `experiments/train_centralized.py`

### Federated Learning (FedAvg)
1. `preprocessing/` → `create_dataset_splits.py` → `create_federated_partitions.py`
2. `federated/run_federated.py` → `federated/server.py` + `federated/client.py`

### Differential Privacy
1. `federated/run_dp.py` → `federated/dp_server.py` + `federated/dp_client.py`

### Membership Inference Attacks
1. `privacy/membership_inference.py` → `privacy/attack_dataset.py`
2. Evaluates against centralized, federated, and DP models

## 3. Important Modules and Responsibilities

**Key Modules:**
- `models/resnet18_model.py`: Core model architecture
- `preprocessing/build_slice_manifest.py`: Slice manifest generation
- `preprocessing/pytorch_dataset.py`: Data loading and preprocessing
- `experiments/train_centralized.py`: Centralized training pipeline
- `federated/server.py`: Federated server logic
- `privacy/membership_inference.py`: MIA implementation

**Configurations:**
- `federated/federated_config.py`: Federated learning parameters
- `federated/dp_config.py`: Differential privacy parameters

**Entry Points:**
- `experiments/train_centralized.py`: Centralized training
- `federated/run_federated.py`: Federated experiments
- `federated/run_dp.py`: DP experiments
- `privacy/membership_inference.py`: MIA experiments

## 4. Dependencies Between Modules

**Import Graph:**
- `experiments/train_centralized.py` → `preprocessing/pytorch_dataset.py`, `models/resnet18_model.py`
- `federated/server.py` → `preprocessing/pytorch_dataset.py`, `models/resnet18_model.py`, `experiments/train_centralized.py`
- `federated/client.py` → `preprocessing/pytorch_dataset.py`, `models/resnet18_model.py`
- `federated/dp_client.py` → `federated/client.py`, `federated/dp_config.py`
- `privacy/membership_inference.py` → `privacy/attack_dataset.py`, `federated/federated_config.py`

**Configuration Dependencies:**
- All experiments depend on `data/partitions/participant_splits.csv`
- All experiments depend on `results/quality_control/extraction_report.csv`
- Federated experiments depend on `data/partitions/federated/` directory

## 5. Duplicate or Redundant Files

**Potential Duplicates:**
1. Empty `__init__.py` files in multiple directories
2. Empty configuration files (`config.yaml`)
3. Empty README.md

**Script Analysis:**
- Several preprocessing scripts may have overlapping functionality:
  - `full_dataset_audit.py` vs `dataset_quality_control.py`
  - `validate_dataset_manifest.py` vs `validate_mri_volumes.py`
- `analyze_inspector.py` and `analyze_*_distribution.py` scripts may be redundant

## 6. Potentially Unused Files

**Questionable Files:**
1. `models/model_utils.py`: Empty file
2. `evaluation/` directory: Empty
3. `attacks/` directory: Empty
4. Empty configuration files

**Scripts Needing Review:**
- `preprocessing/inspect_*` scripts: May be debugging utilities
- `preprocessing/analyze_*` scripts: May be analysis utilities
- `scripts/` directory: May contain debugging/analysis utilities

## 7. Files Recommended for Removal

**Empty/Unused Files:**
1. `models/model_utils.py`
2. `README.md` (empty - will be created in Phase 5)
3. `configs/config.yaml` (empty - needs content)

**Potentially Redundant:**
1. Multiple empty `__init__.py` files
2. `evaluation/` directory (if truly unused)
3. `attacks/` directory (if truly unused)
4. Some `preprocessing/` analysis scripts if they're truly redundant

**Important:** Do not remove any files containing research logic, even if they appear redundant, without thorough investigation.

## 8. Files That Must Be Preserved

**Critical Files (DO NOT DELETE):**
1. **All model weights and checkpoints** in `results/*/*/checkpoints/`
2. **All experiment outputs** in `results/*/` directories
3. **Dataset files** in `data/`:
   - `data/partitions/participant_splits.csv`
   - `data/partitions/federated/*.csv` (client partitions)
   - `data/partitions/slice_manifest.csv`
4. **Extraction report**: `results/quality_control/extraction_report.csv`
5. **Quality control reports**: `results/quality_control/*`
6. **Preprocessing outputs**: `results/preprocessing_analysis/`, `results/quality_control/`
7. **All source code** implementing research logic
8. **Configuration files** (even if empty - they may be populated)
9. **Git history and `.git/` directory**

## 9. Architectural Weaknesses

**Identified Issues:**
1. **Fragmented architecture**: Related functionality spread across many small files
2. **Empty files**: Several empty files that clutter the repository
3. **Missing documentation**: `README.md` is empty
4. **Unclear module boundaries**: Some scripts may have unclear responsibilities
5. **Potential duplication**: Some preprocessing/analysis scripts may duplicate functionality
6. **Inconsistent naming**: Some files use different naming conventions

**Recommendations:**
1. Consolidate related preprocessing scripts
2. Populate `README.md` with comprehensive documentation
3. Populate `config.yaml` with actual configuration
4. Review empty files for actual necessity
5. Establish clear module boundaries

## 10. Recommended Final Architecture

**Proposed Structure:**
```
brain-fl-project/
├── README.md                    # Comprehensive documentation
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore patterns
├── LICENSE                      # Software license
├── configs/                     # Configuration files
│   ├── centralized.yaml
│   ├── federated.yaml
│   └── differential_privacy.yaml
├── data/                        # Dataset
│   ├── raw/                     # Original archives
│   ├── extracted/              # Extracted volumes
│   ├── processed/              # Processed data
│   └── manifests/              # Participant and slice manifests
├── models/                      # Model implementations
│   ├── resnet18_model.py
│   └── utils.py                 # Utility functions
├── preprocessing/               # Data preparation pipeline
│   ├── dataset/                # Dataset preparation
│   ├── slice_extraction/       # Slice extraction
│   ├── validation/             # Data validation
│   └── augmentation/           # Data augmentation (if needed)
├── federated/                   # Federated learning components
│   ├── core/                   # FedAvg implementation
│   ├── dp/                     # Differential privacy
│   └── evaluation/             # Federated evaluation
├── privacy/                     # Privacy analysis
│   ├── membership_inference/   # MIA implementation
│   └── differential_privacy/   # DP analysis
├── experiments/                 # Experiment entry points
│   ├── centralized/           # Centralized experiments
│   └── federated/             # Federated experiments
├── results/                     # Experiment outputs
│   ├── centralized/           # Centralized results
│   ├── federated/             # Federated results
│   ├── differential_privacy/   # DP results
│   └── membership_inference/  # MIA results
├── docs/                        # Documentation
│   ├── REPOSITORY_AUDIT.md
│   ├── CLEANUP_REPORT.md
│   ├── architecture.md
│   └── images/
└── tests/                       # Test suite
```

**Key Improvements:**
1. **Clear separation of concerns**
2. **Consolidated configuration**
3. **Organized results storage**
4. **Comprehensive documentation**
5. **Modular preprocessing pipeline**

## 11. Phase 1 Complete

This audit provides a foundation for repository cleanup and reorganization. The next phase will involve safe cleanup of genuinely unnecessary files while preserving all research-critical components.

**Action Items:**
1. Create `docs/` directory if it doesn't exist
2. Populate `README.md` (Phase 5)
3. Review empty files for actual necessity
4. Consolidate similar preprocessing scripts
5. Establish consistent naming conventions
6. Update imports and paths after restructuring
