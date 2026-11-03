"""Evaluate the selected federated checkpoint on the untouched OASIS-1 test split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.client import _parameters_to_state
from federated.federated_config import FederatedConfig, get_device, set_seed
from models.resnet18_model import build_resnet18
from preprocessing.pytorch_dataset import OASISSliceDataset, SELECTED_SLICES


def participant_predictions(model, loader, device):
    model.eval()
    records = {}
    with torch.no_grad():
        for images, labels, participant_ids in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            for participant_id, label, probability in zip(participant_ids, labels.numpy(), probabilities):
                record = records.setdefault(str(participant_id), {"label": int(label), "probabilities": []})
                if record["label"] != int(label):
                    raise ValueError(f"Inconsistent test label for participant {participant_id}.")
                record["probabilities"].append(float(probability))
    rows = []
    for participant_id, record in records.items():
        if len(record["probabilities"]) != 7:
            raise ValueError(f"Participant {participant_id} has {len(record['probabilities'])} test slices; expected 7.")
        probability = float(np.mean(record["probabilities"]))
        rows.append({
            "participant_id": participant_id,
            "label": record["label"],
            "probability": probability,
            "prediction": int(probability >= 0.5),
            "slice_count": len(record["probabilities"]),
        })
    if len(rows) != 36:
        raise ValueError(f"Expected 36 test participants, found {len(rows)}.")
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def write_plots(predictions, output_dir):
    tn, fp, fn, tp = confusion_matrix(predictions["label"], predictions["prediction"], labels=[0, 1]).ravel()
    cells = [("TN", tn, "#8ecae6"), ("FP", fp, "#e76f51"), ("FN", fn, "#e76f51"), ("TP", tp, "#8ecae6")]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="430">', '<rect width="100%" height="100%" fill="white"/>', '<text x="320" y="32" text-anchor="middle" font-size="22" font-weight="bold">Test confusion matrix</text>']
    svg += ['<text x="320" y="395" text-anchor="middle" font-size="15">Predicted class</text>', '<text x="24" y="220" text-anchor="middle" font-size="15" transform="rotate(-90 24 220)">True class</text>']
    positions = [(180, 95), (360, 95), (180, 250), (360, 250)]
    for (name, value, color), (x, y) in zip(cells, positions):
        svg.append(f'<rect x="{x}" y="{y}" width="150" height="120" fill="{color}" stroke="#333"/>')
        svg.append(f'<text x="{x + 75}" y="{y + 50}" text-anchor="middle" font-size="18">{name}</text>')
        svg.append(f'<text x="{x + 75}" y="{y + 88}" text-anchor="middle" font-size="26" font-weight="bold">{value}</text>')
    svg += ['<text x="255" y="85" text-anchor="middle" font-size="14">CN (0)</text>', '<text x="435" y="85" text-anchor="middle" font-size="14">CDR-positive (1)</text>', '<text x="110" y="170" text-anchor="middle" font-size="14">CN (0)</text>', '<text x="110" y="325" text-anchor="middle" font-size="14">CDR-positive (1)</text>', '</svg>']
    (output_dir / "confusion_matrix.svg").write_text("\n".join(svg) + "\n")

    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(predictions["probability"], bins=bins)
    bars = []
    maximum = max(counts.max(), 1)
    for index, count in enumerate(counts):
        height = count / maximum * 250
        x = 70 + index * 48
        y = 340 - height
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="38" height="{height:.1f}" fill="#2f6690"/>')
        bars.append(f'<text x="{x + 19}" y="{y - 5:.1f}" text-anchor="middle" font-size="11">{count}</text>')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="430">', '<rect width="100%" height="100%" fill="white"/>', '<text x="320" y="32" text-anchor="middle" font-size="22" font-weight="bold">Participant class-1 probabilities</text>', '<line x1="60" y1="340" x2="560" y2="340" stroke="#333"/>', '<line x1="60" y1="70" x2="60" y2="340" stroke="#333"/>'] + bars + ['<text x="310" y="395" text-anchor="middle" font-size="15">Mean class-1 probability</text>', '</svg>']
    (output_dir / "probability_distribution.svg").write_text("\n".join(svg) + "\n")


def evaluate(output_dir: Path):
    config_path = output_dir / "experiment_config.json"
    config_values = json.loads(config_path.read_text()) if config_path.exists() else {}
    config = FederatedConfig(
        batch_size=int(config_values.get("batch_size", 8)),
        seed=int(config_values.get("seed", 42)),
        output_dir=output_dir,
    )
    set_seed(config.seed)
    device = get_device()
    model = build_resnet18(num_classes=2, pretrained=False).to(device)
    checkpoint_path = output_dir / "best_global_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best global checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_dataset = OASISSliceDataset(config.extraction_report, "test", SELECTED_SLICES, config.image_size, 2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    predictions = participant_predictions(model, test_loader, device)
    labels = predictions["label"].to_numpy()
    predicted = predictions["prediction"].to_numpy()
    probabilities = predictions["probability"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    metrics = {
        "checkpoint": str(checkpoint_path), "participants": len(predictions), "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0), "recall": recall_score(labels, predicted, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"), "f1_score": f1_score(labels, predicted, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities), "balanced_accuracy": balanced_accuracy_score(labels, predicted),
        "mcc": matthews_corrcoef(labels, predicted), "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "test_split_used_only_after_training": True,
    }
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    write_plots(predictions, output_dir)
    report = ["Federated FedAvg final evaluation", "", f"Checkpoint: {checkpoint_path}", "Test participants: 36", "Test data was used only after training and model selection.", "", "Metrics:"]
    report.extend(f"{key}: {value}" for key, value in metrics.items() if key != "checkpoint")
    (output_dir / "final_report.txt").write_text("\n".join(report) + "\n")
    print(pd.DataFrame([metrics]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "federated" / "fedavg")
    args = parser.parse_args()
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
