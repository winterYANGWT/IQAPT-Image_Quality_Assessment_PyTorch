from .metric import MultiImageMetric, torch
from . import utils
import torch.nn.functional as F
from .model import fid

__all__ = ['FID']


class FID(MultiImageMetric):
    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.preprocess = fid.PreprocessLayer()
        self.inception = fid.InceptionV3()
        self.inception.eval()

    def calc(self, features_a: torch.Tensor,
             features_b: torch.Tensor) -> torch.Tensor:
        mu_a = torch.mean(features_a, dim=0)
        mu_b = torch.mean(features_b, dim=0)
        sigma_a = torch.cov(features_a.t())
        sigma_b = torch.cov(features_b.t())
        diff_mu = mu_a - mu_b
        covmean, error = utils.sqrtm_newton_schulz(sigma_a.mm(sigma_b),
                                                   num_iters=100)

        if not torch.isfinite(covmean).all():
            offset = utils.to_device(
                torch.eye(sigma_a.size(0), dtype=sigma_a.dtype) * self.eps,
                utils.get_device(sigma_a))
            covmean, error = utils.sqrtm_newton_schulz(
                (sigma_a + offset).mm(sigma_b + offset), num_iters=100)

        fid = diff_mu.dot(diff_mu) + torch.trace(sigma_a + sigma_b -
                                                 2 * covmean)
        return fid

    def calc_features(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images)
        features = self.inception(images)
        return features
