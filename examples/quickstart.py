"""Open a pretrained ResNet-18 on an image from Python.

    python examples/quickstart.py path/to/image.jpg

Downloads ImageNet weights the first time (~45 MB). Without an image it uses a random tensor.
"""

import sys

import torch
import torchvision

import neurotome

weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights).eval()

if len(sys.argv) > 1:
    from PIL import Image

    x = weights.transforms()(Image.open(sys.argv[1]).convert("RGB")).unsqueeze(0)
    name = sys.argv[1]
else:
    x = torch.randn(1, 3, 224, 224)
    name = "random input"

# Captures every op, then serves the UI (Ctrl+C to stop).
neurotome.open(model, x, name=name, categories=weights.meta["categories"])
