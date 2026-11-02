import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_CLASSES = 2
INPUT_CHANNELS = 3


# ============================================================
# RESNET-18 CLASSIFIER
# ============================================================

class OASISResNet18(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=False
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(
            weights=weights
        )

        # Replace the original 1000-class ImageNet head.
        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            input_features,
            num_classes
        )

        self.num_classes = num_classes
        self.pretrained = pretrained

    def forward(self, images):

        if images.ndim != 4:
            raise ValueError(
                "Expected input shape (B, C, H, W), "
                f"received {tuple(images.shape)}"
            )

        if images.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} input channels, "
                f"received {images.shape[1]}"
            )

        return self.backbone(images)


# ============================================================
# MODEL FACTORY
# ============================================================

def build_resnet18(
    num_classes=NUM_CLASSES,
    pretrained=False
):
    """
    Construct the OASIS ResNet-18 classifier.
    """

    model = OASISResNet18(
        num_classes=num_classes,
        pretrained=pretrained
    )

    return model


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_model(model, device):

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(
        2,
        INPUT_CHANNELS,
        224,
        224,
        device=device
    )

    with torch.no_grad():

        output = model(dummy_input)

    assert output.shape == (2, NUM_CLASSES), (
        f"Unexpected output shape: {output.shape}"
    )

    assert torch.isfinite(output).all(), (
        "Model produced NaN or infinite outputs."
    )

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Device:", device)
    print("Model validation: PASSED")

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("OASIS-1 RESNET-18 MODEL VALIDATION")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_resnet18(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTotal parameters:", f"{total_parameters:,}")
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print("Pretrained weights:", model.pretrained)

    model = validate_model(
        model,
        device
    )

    print("\n" + "=" * 65)
    print("RESNET-18 MODEL VALIDATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()