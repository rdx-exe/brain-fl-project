# Federated OASIS-1 Training

This directory implements ordinary four-client FedAvg with Flower 1.42.0 and
the existing PyTorch ResNet-18 pipeline. Clients receive participant-level
files from `data/partitions/federated/`; each client therefore owns all seven
selected slices for its participants. The validation callback uses only the
existing `validation` split and aggregates slice probabilities by participant.

## Environment

Install the project runtime and Flower simulation support in the selected
Python environment:

```bash
python3 -m pip install torch torchvision pandas numpy nibabel scikit-learn
python3 -m pip install "flwr[simulation]==1.42.0"
```

The simulation extra provides Ray, which Flower uses for local virtual clients.

## Run

First generate or validate the four participant partitions:

```bash
python3 preprocessing/create_federated_partitions.py
```

Run a short smoke test:

```bash
python3 federated/run_federated.py \
  --rounds 1 \
  --local-epochs 1 \
  --output-dir /tmp/oasis-federated-smoke
```

Run an experiment with the defaults of five communication rounds:

```bash
python3 federated/run_federated.py --rounds 5 --local-epochs 1
```

Useful options are `--rounds`, `--local-epochs`, `--batch-size`, `--seed`, and
`--output-dir`. CUDA is selected automatically when available; the Flower
simulation reserves one GPU per virtual client when CUDA is available.

## Outputs

The experiment writes to `results/federated/` by default:

- `checkpoints/round_XXX.pth`: global model after each round.
- `checkpoints/best_model.pth`: lowest global validation loss checkpoint.
- `global_validation_history.csv`: participant-level validation metrics.
- `client_X_training_metrics.csv`: local loss, participant count, and slice count per client and round.

The centralized baseline checkpoint under `results/centralized/` is never read
or overwritten. No test samples are loaded by the federated experiment.