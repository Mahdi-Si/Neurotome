"""A custom model loadable by spec: `py:examples.custom_model:build`.

Run from the repo root (so `examples` is importable):

    neurotome capture py:examples.custom_model:build
    # or in the UI's "New run" form, model = py:examples.custom_model:build

A module used this way may define:
  build()        -> nn.Module          (required; the name after the last ':')
  INPUT_SHAPE    list[int]             shape of the random sample, default [1, 3, 224, 224]
  CATEGORIES     list[str]             class names for the output
  preprocess(img: PIL.Image) -> Tensor custom image preprocessing
"""

import torch
from torch import nn

INPUT_SHAPE = [1, 1, 28, 28]
CATEGORIES = [str(i) for i in range(10)]


class Residual(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm = nn.BatchNorm2d(c)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.norm(self.conv1(x))))


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(Residual(16), nn.MaxPool2d(2), Residual(16))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def build() -> nn.Module:
    torch.manual_seed(0)
    return SmallNet()


def preprocess(img):
    from torchvision import transforms

    tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])
    return tf(img).unsqueeze(0)
