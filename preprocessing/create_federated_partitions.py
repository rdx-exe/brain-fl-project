"""Create reproducible participant-level federated learning partitions.

Allocation is always performed on the 164 training participants from
``participant_splits.csv``.  The optional slice manifest is used only to
verify that every allocated participant has all seven training slices and to
write a combined slice-level manifest for downstream data loaders.

The IID strategy distributes each class in a round-robin order while
preferring clients with fewer assigned participants.  The non-IID strategy
draws a class-specific client proportion from a Dirichlet distribution.  A
draw is retried until every client satisfies ``min_client_participants``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTICIPANT_MANIFEST = PROJECT_ROOT / "data" / "partitions" / "participant_splits.csv"
DEFAULT_SLICE_MANIFEST = PROJECT_ROOT / "data" / "partitions" / "slice_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "partitions" / "federated"
EXPECTED_CLIENTS = 4
EXPECTED_SLICES_PER_PARTICIPANT = 7
EXPECTED_SPLITS = {"train", "validation", "test"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("iid", "dirichlet"), default="iid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration for non-IID allocation.")
    parser.add_argument("--min-client-participants", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=EXPECTED_CLIENTS)
    parser.add_argument("--participant-manifest", type=Path, default=DEFAULT_PARTICIPANT_MANIFEST)
    parser.add_argument("--slice-manifest", type=Path, default=DEFAULT_SLICE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_manifests(participant_path, slice_path):
    participant_df = pd.read_csv(participant_path)
    required = {"participant_id", "label", "label_name", "split"}
    missing = required - set(participant_df.columns)
    if missing:
        raise ValueError(f"Participant manifest is missing columns: {sorted(missing)}")
    if participant_df["participant_id"].duplicated().any():
        raise ValueError("Participant manifest must contain one row per participant.")
    if participant_df["label"].isna().any() or participant_df["split"].isna().any():
        raise ValueError("Participant labels and split assignments must be present.")
    if not set(participant_df["split"]) <= EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split values: {sorted(set(participant_df['split']) - EXPECTED_SPLITS)}")

    slice_df = pd.read_csv(slice_path)
    required_slice = {"participant_id", "split", "label", "slice_index"}
    missing = required_slice - set(slice_df.columns)
    if missing:
        raise ValueError(f"Slice manifest is missing columns: {sorted(missing)}")
    return participant_df, slice_df


def allocate_iid(training_df, num_clients, rng):
    assignments = {client: [] for client in range(1, num_clients + 1)}
    client_sizes = np.zeros(num_clients, dtype=int)
    for label in sorted(training_df["label"].unique()):
        ids = training_df.loc[training_df["label"] == label, "participant_id"].to_numpy()
        rng.shuffle(ids)
        order = np.argsort(client_sizes, kind="stable")
        for offset, participant_id in enumerate(ids):
            client_index = order[offset % num_clients]
            assignments[client_index + 1].append(participant_id)
            client_sizes[client_index] += 1
            order = np.argsort(client_sizes, kind="stable")
    return assignments


def allocate_dirichlet(training_df, num_clients, alpha, minimum, rng):
    if alpha <= 0:
        raise ValueError("--alpha must be greater than zero for Dirichlet allocation.")
    if minimum < 1:
        raise ValueError("--min-client-participants must be at least one.")
    if minimum * num_clients > len(training_df):
        raise ValueError("Minimum client requirement exceeds the number of training participants.")

    label_groups = {
        label: training_df.loc[training_df["label"] == label, "participant_id"].to_numpy()
        for label in sorted(training_df["label"].unique())
    }
    for _ in range(1000):
        assignments = {client: [] for client in range(1, num_clients + 1)}
        for label, ids in label_groups.items():
            probabilities = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
            counts = rng.multinomial(len(ids), probabilities)
            shuffled_ids = ids.copy()
            rng.shuffle(shuffled_ids)
            start = 0
            for client_index, count in enumerate(counts):
                assignments[client_index + 1].extend(shuffled_ids[start:start + count])
                start += count
        if min(len(ids) for ids in assignments.values()) >= minimum:
            return assignments
    raise RuntimeError("Could not satisfy --min-client-participants after 1000 Dirichlet draws.")


def validate_partitions(participant_df, slice_df, client_frames, num_clients):
    training_ids = set(participant_df.loc[participant_df["split"] == "train", "participant_id"])
    held_out_ids = set(participant_df.loc[participant_df["split"] != "train", "participant_id"])
    client_sets = [set(frame["participant_id"]) for frame in client_frames.values()]
    assigned_ids = set().union(*client_sets)
    overlap = set().union(*(client_sets[i] & client_sets[j] for i in range(num_clients) for j in range(i + 1, num_clients)))
    if assigned_ids != training_ids:
        raise ValueError(f"Client union differs from training set: missing={sorted(training_ids - assigned_ids)}, extra={sorted(assigned_ids - training_ids)}")
    if overlap:
        raise ValueError(f"Participant overlap between clients: {sorted(overlap)}")
    held_out_overlap = assigned_ids & held_out_ids
    if held_out_overlap:
        raise ValueError(f"Validation/test participants assigned to clients: {sorted(held_out_overlap)}")

    expected_labels = participant_df.set_index("participant_id")["label"]
    for client_id, frame in client_frames.items():
        if not frame["participant_id"].map(expected_labels).equals(frame["label"]):
            raise ValueError(f"Label mismatch detected in client {client_id}.")
        client_slices = slice_df[slice_df["participant_id"].isin(frame["participant_id"])]
        counts = client_slices.groupby("participant_id")["slice_index"].nunique()
        if len(counts) != len(frame) or not (counts == EXPECTED_SLICES_PER_PARTICIPANT).all():
            raise ValueError(f"Client {client_id} does not contain exactly seven unique slices per participant.")
        if not (client_slices["split"] == "train").all():
            raise ValueError(f"Non-training slice found in client {client_id}.")

    return {
        "training_participants": len(training_ids),
        "assigned_participants": len(assigned_ids),
        "client_overlap_count": len(overlap),
        "held_out_overlap_count": len(held_out_overlap),
        "all_training_participants_assigned": True,
        "validation_test_excluded": True,
        "seven_slices_per_participant": True,
    }


def distribution_report(client_frames, strategy, seed, alpha):
    rows = []
    for client_id, frame in client_frames.items():
        counts = frame["label_name"].value_counts()
        rows.append({
            "client_id": f"client_{client_id}",
            "strategy": strategy,
            "seed": seed,
            "dirichlet_alpha": alpha if strategy == "dirichlet" else "",
            "participant_count": len(frame),
            "slice_count": len(frame) * EXPECTED_SLICES_PER_PARTICIPANT,
            "CN_count": int(counts.get("CN", 0)),
            "CDR_positive_count": int(counts.get("CDR_positive", 0)),
            "CN_percentage": round(counts.get("CN", 0) / len(frame) * 100, 2),
            "CDR_positive_percentage": round(counts.get("CDR_positive", 0) / len(frame) * 100, 2),
        })
    return pd.DataFrame(rows)


def write_visualizations(report, output_dir):
    def write_svg(path, title, series):
        width, height = 720, 420
        chart_left, chart_top, chart_width, chart_height = 80, 55, 590, 290
        maximum = max(max(values) for _, values, _ in series) or 1
        bar_width = chart_width / (len(report) * len(series) + len(report) + 1)
        bars = []
        labels = []
        for client_index, client_id in enumerate(report["client_id"]):
            group_start = chart_left + bar_width * (client_index * (len(series) + 1) + 1)
            labels.append(f'<text x="{group_start + bar_width * len(series) / 2:.1f}" y="375" text-anchor="middle" font-size="14">{client_id}</text>')
            for series_index, (_, values, color) in enumerate(series):
                value = values[client_index]
                bar_height = value / maximum * chart_height
                x = group_start + series_index * bar_width
                y = chart_top + chart_height - bar_height
                bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 3:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
                bars.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="11">{value}</text>')
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
        svg.append(f'<rect width="100%" height="100%" fill="white"/><text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">{title}</text>')
        svg.append(f'<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#333"/>')
        svg.append(f'<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" stroke="#333"/>')
        svg.extend(bars)
        svg.extend(labels)
        for index, (name, _, color) in enumerate(series):
            x = chart_left + index * 150
            svg.append(f'<rect x="{x}" y="395" width="14" height="14" fill="{color}"/><text x="{x + 20}" y="407" font-size="12">{name}</text>')
        svg.append('</svg>')
        path.write_text("\n".join(svg) + "\n")

    write_svg(
        output_dir / "participant_distribution.svg",
        "Participant count by client",
        [("Participants", report["participant_count"].tolist(), "#2f6690")],
    )
    write_svg(
        output_dir / "class_distribution.svg",
        "Class distribution by client",
        [("CN", report["CN_count"].tolist(), "#8ecae6"), ("CDR positive", report["CDR_positive_count"].tolist(), "#e76f51")],
    )


def main():
    args = parse_args()
    if args.num_clients != EXPECTED_CLIENTS:
        raise ValueError("This project configuration requires exactly four clients.")
    participant_df, slice_df = load_manifests(args.participant_manifest, args.slice_manifest)
    training_df = participant_df[participant_df["split"] == "train"].copy()
    if len(training_df) != 164:
        raise ValueError(f"Expected 164 training participants, found {len(training_df)}.")
    training_slice_df = slice_df[slice_df["split"] == "train"]
    if len(training_slice_df) != len(training_df) * EXPECTED_SLICES_PER_PARTICIPANT:
        raise ValueError("Training slice manifest does not contain seven slices for every training participant.")

    rng = np.random.default_rng(args.seed)
    if args.strategy == "iid":
        assignments = allocate_iid(training_df, args.num_clients, rng)
    else:
        assignments = allocate_dirichlet(training_df, args.num_clients, args.alpha, args.min_client_participants, rng)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    client_frames = {}
    for client_id, participant_ids in assignments.items():
        frame = training_df[training_df["participant_id"].isin(participant_ids)].copy()
        frame["client_id"] = f"client_{client_id}"
        frame["partition_strategy"] = args.strategy
        frame["random_seed"] = args.seed
        frame = frame.sort_values("participant_id").reset_index(drop=True)
        client_frames[client_id] = frame

    validation = validate_partitions(participant_df, slice_df, client_frames, args.num_clients)
    report = distribution_report(client_frames, args.strategy, args.seed, args.alpha)
    for client_id, frame in client_frames.items():
        frame.to_csv(output_dir / f"client_{client_id}.csv", index=False)
    combined_slice = pd.concat(
        [training_slice_df[training_slice_df["participant_id"].isin(frame["participant_id"])]
         .assign(client_id=f"client_{client_id}") for client_id, frame in client_frames.items()],
        ignore_index=True,
    )
    combined_slice.to_csv(output_dir / "client_slice_manifest.csv", index=False)
    report.to_csv(output_dir / "client_distribution.csv", index=False)
    pd.DataFrame([{"check": key, "value": value} for key, value in validation.items()]).to_csv(
        output_dir / "partition_validation_report.csv", index=False
    )
    config = {
        "strategy": args.strategy,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "dirichlet_alpha": args.alpha,
        "min_client_participants": args.min_client_participants,
        "participant_manifest": str(args.participant_manifest),
        "slice_manifest": str(args.slice_manifest),
        "expected_slices_per_participant": EXPECTED_SLICES_PER_PARTICIPANT,
    }
    (output_dir / "reproducibility_config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_visualizations(report, output_dir)
    print(report.to_string(index=False))
    print(f"\nValidated partition files written to: {output_dir}")


if __name__ == "__main__":
    main()