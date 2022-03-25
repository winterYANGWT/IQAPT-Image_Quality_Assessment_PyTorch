from .metric import SingleImageMetric, torch
from .. import utils
from ..model import lpips
import einops

__all__ = ['LPIPS']


class LPIPS(SingleImageMetric):
    def __init__(self,
                 net='vgg',
                 linear_calibration=True,
                 spatial=False) -> None:
        super().__init__()
        self.linear_calibration = linear_calibration
        self.spatial = spatial
        self.feature_extractor = lpips.feature_extractors[net]()
        self.scaling_layer = lpips.ScalingLayer()
        self.feature_extractor = self.feature_extractor.to(torch.float64)
        self.scaling_layer = self.scaling_layer.to(torch.float64)

        if linear_calibration == True:
            self.linear_layer = lpips.LinearCalibrationLayer(net)
            self.linear_layer = self.linear_layer.to(torch.float64)

    def calc(self, images_a: torch.Tensor,
             images_b: torch.Tensor) -> torch.Tensor:
        features_a = self.calc_features(images_a)
        features_b = self.calc_features(images_b)
        diffs = [(feature_a - feature_b)**2
                 for (feature_a, feature_b) in zip(features_a, features_b)]

        if self.linear_calibration == True:
            percepts = self.linear_layer(diffs)
        else:
            percepts = [diff.sum(dim=1, keepdim=True) for diff in diffs]

        if self.spatial == True:
            lpips = [
                utils.upsample(percept, out_size=images_a.shape[2:])
                for percept in percepts
            ]
        else:
            lpips = [
                utils.spatial_average(percept, keepdim=True)
                for percept in percepts
            ]

        lpips = sum(lpips)
        lpips = einops.rearrange(lpips, 'N 1 1 1-> N')
        return lpips

    def calc_features(self, images: torch.Tensor) -> torch.Tensor:
        images = self.scaling_layer(images)
        features = self.feature_extractor(images)
        features = [utils.normalize_tensor(feature) for feature in features]
        return features


# based on https://github.com/richzhang/PerceptualSimilarity
# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
# All rights reserved.
