import torch
import torchvision

from torch import nn


def create_effnetb0_model_deployment(num_classes: int = 3,
                                     seed: int = 42,
                                     device: torch.device = 'cpu'):
    """Creates an EfficientNetB0 feature extractor model and transforms.

    Args:
        device: the device to put the model on. Defaults to cpu.
        num_classes (int, optional): number of classes in the classifier head. Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB0 feature extractor model.
        transforms (torchvision.transforms): EffNetB0 image transforms.
    """
    # 1, 2, 3. Create EffNetB0 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 4. Freeze all layers in the base model
    for param in model.parameters():
        param.requires_grad = False

    # 5. Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=num_classes)
    ).to(device)

    # 6. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new a {model.name} model for deployment.")

    return model, transforms
