"""FoodVision Big: EfficientNet-B2 image classifier for the 101 dishes of Food-101.

Single source of truth for the architecture, preprocessing and inference. It is used by
the Hugging Face Space (`space/app.py`), by the training code (`training/src/`), by the
Inference Endpoint (`handler.py`) and by anyone who downloads this repo from the Hub:

    import model
    predictor = model.load("path/to/this/repo", device="cpu")
    predictor.predict("dish.jpg")            # -> {"apple_pie": 0.001, ..., "sushi": 0.93, ...} (all 101)
    predictor.predict("dish.jpg", top_k=5)   # -> the 5 most likely dishes only
"""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

REPO_ID = "shalev396/foodvision-big"
FRAMEWORK = "pytorch"
WEIGHTS_FILE = "model.safetensors"

# ImageNet weights the backbone is initialised from for training. Their preset also defines the
# preprocessing. Same object as `EfficientNet_B2_Weights.DEFAULT`, pinned so it can never drift.
BACKBONE_WEIGHTS = EfficientNet_B2_Weights.IMAGENET1K_V1


def cuda_available() -> bool:
    return torch.cuda.is_available()


def get_transform():
    """PIL image -> normalized (3, 288, 288) tensor, exactly as EfficientNet-B2 was pretrained:
    resize to 288 (bicubic) -> center-crop 288 -> ImageNet mean/std.

    Used for evaluation and inference. Training puts `TrivialAugmentWide` in front of it
    (see `training/src/data_setup.py`).
    """
    return BACKBONE_WEIGHTS.transforms()


class FoodVisionNet(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["ml-lab", "image-classification"],
    repo_url="https://github.com/shalev396/ml-lab",
    pipeline_tag="image-classification",
    license="mit",
):
    """torchvision EfficientNet-B2 whose 1000-class ImageNet head is replaced by
    `Dropout(dropout) -> Linear(1408, len(class_names))`.

    Args:
        class_names: output labels in logit order (written to config.json by the mixin).
        dropout: dropout probability in front of the linear head.
        pretrained: initialise the backbone from ImageNet weights. Only needed to *train*;
            `load()` always passes `pretrained=False`, so inference never downloads anything.
    """

    def __init__(self, class_names: list[str], dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        self.class_names = list(class_names)
        self.backbone = efficientnet_b2(weights=BACKBONE_WEIGHTS if pretrained else None)
        in_features = self.backbone.classifier[1].in_features  # 1408 for B2
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, len(self.class_names)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) normalized images -> (N, num_classes) logits."""
        return self.backbone(x)


class Predictor:
    """Loads the trained weights once and classifies single images."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        model_dir = Path(model_dir)
        if not (model_dir / WEIGHTS_FILE).is_file():
            raise FileNotFoundError(
                f"{WEIGHTS_FILE} not found in {model_dir}: train/export the model first "
                f"or download it with huggingface_hub.snapshot_download('{REPO_ID}')."
            )
        self.device = torch.device(device)
        self.model = FoodVisionNet.from_pretrained(
            model_dir, pretrained=False, map_location="cpu", strict=True
        )
        self.model.to(self.device).eval()
        self.class_names = self.model.class_names
        self.transform = get_transform()

    @torch.inference_mode()
    def predict(self, image: Image.Image | str | Path, top_k: int | None = None) -> dict[str, float]:
        """Image (PIL or file path) -> {class_name: probability}.

        By default ALL 101 classes are returned (they sum to 1). `top_k=5` returns only the
        5 most likely classes, highest first.
        """
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return self.predict(img.convert("RGB"), top_k=top_k)
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0].float().cpu().tolist()
        result = dict(zip(self.class_names, probs))
        if top_k:
            result = dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k)])
        return result


def load(model_dir: str | Path, device: str = "cpu") -> Predictor:
    return Predictor(model_dir, device)
