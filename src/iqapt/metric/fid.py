from .metric import MultiImageMetric, torch
from .. import utils
from ..model import fid
import einops

__all__ = ['FID']


class FID(MultiImageMetric):
    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.preprocess = fid.PreprocessLayer()
        self.inception = fid.InceptionV3()
        self.preprocess = self.preprocess.to(torch.float64)
        self.inception = self.inception.to(torch.float64)
        self.inception.eval()

    def calc(self, features_a: torch.Tensor,
             features_b: torch.Tensor) -> torch.Tensor:
        mu_a, sigma_a = self.calc_mu_and_sigma(features_a)
        mu_b, sigma_b = self.calc_mu_and_sigma(features_b)
        diff_mu = mu_a - mu_b
        covmean, error = self.calc_covmean(sigma_a, sigma_b)
        fid = diff_mu.dot(diff_mu) + torch.trace(sigma_a + sigma_b -
                                                 2 * covmean)
        return fid

    def calc_features(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images)
        features = self.inception(images)
        return features

    def calc_mu_and_sigma(self, features: torch.Tensor) -> torch.Tensor:
        mu = einops.reduce(features, 'N C->C', 'mean')
        sigma = torch.cov(features.t())
        return mu, sigma

    def calc_covmean(self, sigma_a: torch.Tensor,
                     sigma_b: torch.Tensor) -> torch.Tensor:
        sigma_ab = sigma_a @ sigma_b
        covmean, error = utils.sqrtm_newton_schulz(sigma_ab, num_iters=100)

        if not torch.isfinite(covmean).all():
            offset = torch.eye(sigma_ab.shape,
                               dtype=sigma_ab.dtype,
                               device=sigma_ab.device) * self.eps
            sigma_ab = (sigma_a + offset) @ (sigma_b + offset)
            covmean, error = utils.sqrtm_newton_schulz(sigma_ab, num_iters=100)

        return covmean, error
