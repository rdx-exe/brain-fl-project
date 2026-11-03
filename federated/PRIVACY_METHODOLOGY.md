# Participant-Level Differential Privacy Methodology

## Privacy unit

The protected unit is an OASIS-1 participant, not an MRI slice. Each
participant contributes seven correlated slices. The DP data loader therefore
contains one item per participant, with all seven slices grouped together.
The model averages the seven slice logits before calculating the participant
loss. Opacus sees one logical training example and produces one per-participant
gradient for clipping and noise addition.

## Mechanism

The DP client uses Opacus `PrivacyEngine` with DP-SGD:

1. Sample participant units with Poisson sampling.
2. Compute per-participant gradients.
3. Clip each participant gradient to `max_grad_norm`.
4. Add Gaussian noise with standard deviation
   `noise_multiplier * max_grad_norm` to the clipped aggregate.
5. Apply AdamW to the noised aggregate.
6. Account for each optimizer step with Opacus' RDP accountant.

The configured privacy parameters are `noise_multiplier`, `max_grad_norm`,
participant sampling rate, and `delta`. Epsilon is reported for the configured
delta after every local private step and written to the per-client DP metrics.
The accountant composes steps across local epochs and repeated Flower rounds
within the client process.

## Scope and threat model

This implementation provides a local-training/update-level participant DP
claim under these assumptions:

- The client device trusts its local data loader, model execution, and random
  noise generation.
- The server may observe the client model update and does not receive raw
  participant slices.
- The participant-level DP guarantee applies to the client update produced by
  the DP-SGD mechanism, relative to adding or removing one participant from
  that client's local participant dataset, subject to the accountant's
  sampling and composition assumptions.

Flower FedAvg itself is only an aggregation protocol. It does not provide DP,
and this implementation does not claim central-model DP merely because FedAvg
is used. The server sees individual client updates unless an additional secure
aggregation protocol is added.

The implementation is not a central trusted-aggregator guarantee, does not
provide secure aggregation, and does not protect against a malicious client
that ignores the protocol. Client-level privacy across the entire federated
population also requires an explicit composition/accounting policy for the
number of client updates and the server's view; this implementation logs local
client accounting and makes that scope explicit rather than silently claiming
end-to-end central DP.

## Architectural implication

The non-private pipeline uses torchvision BatchNorm. BatchNorm is invalid for
Opacus per-sample accounting, so the DP path replaces it with GroupNorm using
`ModuleValidator.fix`. This is a necessary privacy compatibility change and
means DP results must be compared against a matching GroupNorm non-private
control if scientific model comparisons are required.

## What has not been done

No final DP OASIS experiment has been run. Before doing so, the privacy budget,
delta, noise multiplier, max norm, number of rounds, local epochs, participant
sampling rate, and threat-model scope must be reviewed and approved. DP
accounting output should be inspected for every client and round before making
an empirical privacy claim.
