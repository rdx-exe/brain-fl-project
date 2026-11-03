# Differential Privacy Compatibility Report

## Runtime check

- PyTorch: `2.15.0+cu132`
- Opacus: `1.6.0`
- Existing optimizer: `torch.optim.AdamW`
- Existing model: torchvision ResNet-18 with a two-class linear head
- Existing input: three-channel, 224 x 224 MRI slices

## Existing model validation

`ModuleValidator.validate(build_resnet18(pretrained=False), strict=False)`
reported 20 BatchNorm-related errors. The model contains BatchNorm layers in
its torchvision ResNet-18 residual blocks. BatchNorm is not compatible with
per-example DP gradients because a sample's normalization depends on other
samples in the batch.

`ModuleValidator.fix` replaces the BatchNorm modules with GroupNorm and the
fixed model has zero remaining Opacus validation errors. The DP path therefore
keeps the ResNet-18 convolutional/head structure but uses privacy-safe GroupNorm
instead of BatchNorm. The ordinary centralized and non-private federated models
are unchanged.

The synthetic Opacus step also exposed torchvision's in-place ReLU and residual
addition operations as backward-hook conflicts. The DP path disables in-place
ReLU and uses an out-of-place residual addition in `BasicBlock`; this preserves
the weights and mathematical operation while making autograd compatible with
Opacus.

## Privacy-unit validation

The existing client loader exposes 1,148 slice examples. Applying Opacus to that
loader would clip and noise slice-level gradients and would not establish
participant-level protection because seven correlated slices from one person
could contribute seven separate privacy units.

The DP client instead exposes exactly one logical sample per participant. Each
item has shape `(7, 3, 224, 224)`, the model computes all seven slice logits and
averages them, and the participant-level loss is passed to Opacus. Thus one
participant contributes one per-sample gradient, which is the unit clipped and
noised by DP-SGD.

## Validation status

- Opacus validator: passed after BatchNorm to GroupNorm conversion.
- Synthetic participant-batch DP-SGD step: passed after the in-place operation fixes.
- Participant unit: explicitly enforced by `ParticipantDataset`.
- Seven-slice grouping: explicitly enforced and checked.
- Poisson sampling: required by `DPConfig` and passed to `PrivacyEngine`.
- Final privacy experiment: **not run by this integration task**.
