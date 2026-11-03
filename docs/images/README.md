# Image Checklist: Brain FL Project README

This document provides a complete checklist of all images referenced in the `README.md` file, along with instructions for their creation, source, and exact destination paths.

## Image Checklist

| Image | Purpose | Source | Required Action |
|---|---|---|---|
| `docs/images/project-banner.png` | Project header banner | Original project graphic | Create manually |
| `docs/images/dataset-pipeline.png` | MRI preprocessing workflow | Actual preprocessing pipeline | Generate or create manually |
| `docs/images/system-architecture.png` | Overall system architecture | Actual repository architecture | Generate or create manually |
| `docs/images/centralized-training-history.png` | Training curves | Existing training logs | Copy existing plot or generate |
| `docs/images/confusion-matrix.png` | Classification performance | Test evaluation output | Copy existing plot (`results/federated/fedavg/confusion_matrix.svg`) |
| `docs/images/centralized-roc-curve.png` | ROC curve | Evaluation output | Copy existing plot |
| `docs/images/centralized-pr-curve.png` | Precision-recall curve | Evaluation output | Copy existing plot |
| `docs/images/federated-client-distribution.png` | Client participant distribution | Partition outputs | Copy existing plot (`data/partitions/federated/participant_distribution.svg`) |
| `docs/images/fedavg-convergence.png` | Federated convergence curves | FedAvg validation history | Copy existing plot |
| `docs/images/iid-vs-noniid.png` | IID vs non-IID comparison | Comparison outputs | Copy existing plot (`results/federated/comparison/`) |
| `docs/images/dp-privacy-utility.png` | Privacy–utility trade-off | DP experiment outputs | Generate or copy existing |
| `docs/images/mia-comparison.png` | Membership inference attack ROC/metrics | MIA outputs | Copy existing plot (`results/privacy/mia/roc_comparison.svg`) |
| `docs/images/final-comparison.png` | Overall comparative analysis | Final report outputs | Generate or compile from results |

## Detailed Image Specifications

### 1. Project Banner (`docs/images/project-banner.png`)
- **Purpose**: Visual banner at the top of the README.
- **Contents**: A research-themed graphic illustrating brain MRI analysis, federated client nodes, a central aggregation server, and privacy-preserving elements.
- **Dimensions**: 1600 × 500 pixels.
- **Action**: Create an original project graphic and save to the specified path.

### 2. Dataset Pipeline (`docs/images/dataset-pipeline.png`)
- **Purpose**: Illustrate the end-to-end dataset preprocessing pipeline.
- **Contents**: Raw MRI archives → volume extraction → mask application → slice selection (7 axial slices) → intensity normalization → resizing (224×224) → participant-level manifests and train/val/test partitions.
- **Dimensions**: 1200 × 600 pixels.
- **Action**: Create diagram using a tool like Mermaid, Excalidraw, or matplotlib, and save to the specified path.

### 3. System Architecture (`docs/images/system-architecture.png`)
- **Purpose**: High-level overview of the federated learning and privacy architecture.
- **Contents**: OASIS-1 dataset, client nodes, Flower server, FedAvg aggregation, Opacus differential privacy, centralized baseline, and membership inference evaluation.
- **Dimensions**: 1400 × 700 pixels.
- **Action**: Create system architecture diagram and save to the specified path.

### 4. Evaluation and Experiment Plots
- **Source**: Most experimental plots already exist as SVG files generated during experiment runs in `results/` and `data/partitions/`.
- **Action**: Convert existing `.svg` files to `.png` where necessary for Markdown compatibility, or link the `.svg` files directly.
  - `confusion-matrix.png`: Copy from `results/federated/fedavg/confusion_matrix.svg` (or convert to PNG).
  - `federated-client-distribution.png`: Copy from `data/partitions/federated/participant_distribution.svg`.
  - `iid-vs-noniid.png`: Copy from `results/federated/comparison/validation_loss_comparison.svg`.
  - `mia-comparison.png`: Copy from `results/privacy/mia/roc_comparison.svg`.

## Notes for Contributors
- All image links in `README.md` use relative paths under `docs/images/`.
- If an image file is missing when viewing the README, an explanatory HTML comment placeholder is provided in the Markdown source.
