"""Compare controlled IID and Dirichlet non-IID FedAvg experiments."""

import argparse
import json
from pathlib import Path

import pandas as pd


def write_svg(path, title, series, y_label):
    width, height = 760, 440
    left, top, chart_width, chart_height = 75, 55, 630, 300
    all_values = [value for _, values, _ in series for value in values]
    lower = min(all_values)
    upper = max(all_values)
    spread = upper - lower or 1.0
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="380" y="30" text-anchor="middle" font-size="21" font-weight="bold">{title}</text>', f'<text x="20" y="220" text-anchor="middle" font-size="14" transform="rotate(-90 20 220)">{y_label}</text>', f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#333"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333"/>']
    for name, values, color in series:
        points = []
        for index, value in enumerate(values):
            x = left + index * chart_width / max(len(values) - 1, 1)
            y = top + chart_height - (value - lower) / spread * chart_height
            points.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
    for index in range(len(series[0][1])):
        x = left + index * chart_width / max(len(series[0][1]) - 1, 1)
        lines.append(f'<text x="{x:.1f}" y="380" text-anchor="middle" font-size="12">{index}</text>')
    lines.append('<text x="380" y="410" text-anchor="middle" font-size="14">Communication round</text>')
    for index, (name, _, color) in enumerate(series):
        x = 120 + index * 190
        lines.append(f'<line x1="{x}" y1="430" x2="{x + 25}" y2="430" stroke="{color}" stroke-width="3"/><text x="{x + 32}" y="435" font-size="13">{name}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n")


def compare(iid_dir: Path, non_iid_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {"iid": iid_dir, "non_iid": non_iid_dir}
    histories = {name: pd.read_csv(path / "round_history.csv") for name, path in names.items()}
    distributions = {}
    for name, path in names.items():
        config = json.loads((path / "experiment_config.json").read_text())
        partition_dir = Path(config["partition_dir"])
        distributions[name] = pd.read_csv(partition_dir / "client_distribution.csv")
    client_metrics = {name: pd.read_csv(path / "client_metrics.csv") for name, path in names.items()}
    test_metrics = {name: pd.read_csv(path / "test_metrics.csv").iloc[0] for name, path in names.items()}

    colors = {"iid": "#2f6690", "non_iid": "#e76f51"}
    write_svg(output_dir / "validation_loss_comparison.svg", "IID vs non-IID validation loss", [(name, frame["val_loss"].tolist(), colors[name]) for name, frame in histories.items()], "Validation loss")
    write_svg(output_dir / "validation_accuracy_comparison.svg", "IID vs non-IID validation accuracy", [(name, frame["val_accuracy"].tolist(), colors[name]) for name, frame in histories.items()], "Accuracy")

    distribution_table = pd.concat(distributions.values(), ignore_index=True)
    distribution_table.to_csv(output_dir / "client_class_distributions.csv", index=False)
    distribution_series = []
    for name, frame in distributions.items():
        distribution_series.append((f"{name} CDR-positive %", frame["CDR_positive_percentage"].tolist(), colors[name]))
    write_svg(output_dir / "client_class_distribution_comparison.svg", "Client CDR-positive distribution", distribution_series, "CDR-positive percentage")
    rows = []
    for name in names:
        frame = histories[name]
        local = client_metrics[name]
        final_local = local[local["round"] == local["round"].max()]
        best_index = frame["val_loss"].idxmin()
        best = frame.loc[best_index]
        rows.append({
            "experiment": name,
            "best_round": int(best["round"]),
            "best_validation_loss": float(best["val_loss"]),
            "best_validation_accuracy": float(best["val_accuracy"]),
            "best_validation_auc": float(best["val_auc"]),
            "test_accuracy": float(test_metrics[name]["accuracy"]),
            "test_f1_score": float(test_metrics[name]["f1_score"]),
            "test_roc_auc": float(test_metrics[name]["roc_auc"]),
            "test_log_loss": float(test_metrics[name]["log_loss"]),
            "final_client_train_loss_mean": float(final_local["train_loss"].mean()),
            "final_client_train_loss_std": float(final_local["train_loss"].std(ddof=0)),
            "final_client_train_accuracy_std": float(final_local["train_accuracy"].std(ddof=0)),
            "client_class_percentage_std": float(distributions[name]["CDR_positive_percentage"].std(ddof=0)),
            "validation_loss_std_after_initial": float(frame.loc[frame["round"] > 0, "val_loss"].std(ddof=0)),
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "final_metric_comparison.csv", index=False)

    iid_row = comparison[comparison["experiment"] == "iid"].iloc[0]
    non_iid_row = comparison[comparison["experiment"] == "non_iid"].iloc[0]
    report = [
        "IID versus non-IID FedAvg analysis", "",
        "Design: Both experiments used the same ResNet-18 architecture, random initialization, preprocessing, 164 training participants, 35 validation participants, 36 test participants, four clients, five rounds, one local epoch, AdamW (lr=1e-4, weight_decay=1e-4), and seed 42.",
        "The non-IID condition used class-conditional Dirichlet allocation with alpha=0.3 and a minimum of 10 participants per client.",
        "Test data were evaluated only after validation-loss model selection and were not used for tuning.", "",
        f"IID best validation loss: {iid_row['best_validation_loss']:.6f} at round {int(iid_row['best_round'])}.",
        f"Non-IID best validation loss: {non_iid_row['best_validation_loss']:.6f} at round {int(non_iid_row['best_round'])}.",
        f"IID test accuracy/F1/ROC-AUC: {iid_row['test_accuracy']:.4f}/{iid_row['test_f1_score']:.4f}/{iid_row['test_roc_auc']:.4f}.",
        f"Non-IID test accuracy/F1/ROC-AUC: {non_iid_row['test_accuracy']:.4f}/{non_iid_row['test_f1_score']:.4f}/{non_iid_row['test_roc_auc']:.4f}.", "",
        "Interpretation:",
        "Client heterogeneity is quantified by the standard deviation of client CDR-positive percentages and by variation in final local training loss and accuracy. Larger values indicate less uniform local objectives.",
        "Convergence is assessed from the round of minimum validation loss and the validation-loss standard deviation after the initial model. These are validation-only criteria; the test metrics are descriptive final evaluations and do not determine the selected round.",
        "Because this is a single fixed seed and one split, observed differences are experimental evidence for this run rather than a population-level claim. Repeated seeds would be needed for uncertainty estimates.",
    ]
    (output_dir / "scientific_analysis.txt").write_text("\n".join(report) + "\n")
    print(comparison.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iid-dir", type=Path, default=Path("results/federated/iid"))
    parser.add_argument("--non-iid-dir", type=Path, default=Path("results/federated/non_iid"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/federated/comparison"))
    args = parser.parse_args()
    compare(args.iid_dir, args.non_iid_dir, args.output_dir)


if __name__ == "__main__":
    main()
