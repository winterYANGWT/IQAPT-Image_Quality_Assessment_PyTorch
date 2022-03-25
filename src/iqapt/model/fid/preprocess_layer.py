import torch.nn.functional as F
import torch.nn as nn


class PreprocessLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, images):
        images = F.interpolate(images,
                               size=(299, 299),
                               mode='bilinear',
                               align_corners=False)
        images = 2 * images - 1
        return images
