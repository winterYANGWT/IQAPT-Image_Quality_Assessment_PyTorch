import torch.nn.functional as F
from .metric import SingleImageMetric, torch
from .. import utils
import einops

__all__ = ['MSSIM']


class SSIM(object):
    def __init__(self) -> None:
        super().__init__()


class MSSIM(SingleImageMetric):
    def __init__(self,
                 window_size=None,
                 K1=0.01,
                 K2=0.03,
                 sigma=1.5,
                 L=1.0,
                 downsample=True) -> None:
        super().__init__()
        self.downsample = downsample
        truncate = 3.5

        if window_size == None:
            r = round(truncate * sigma)
            window_size = 2 * r + 1
        else:
            window_size = window_size

        self.gauss_filter = utils.filter.GaussianFilter(window_size, sigma)
        self.gauss_filter = self.gauss_filter.to(torch.float64)
        NP = window_size**2
        self.conv_norm = NP / (NP - 1)
        self.C1 = (K1 * L)**2
        self.C2 = (K2 * L)**2

    def calc(self, images_a: torch.Tensor,
             images_b: torch.Tensor) -> torch.Tensor:
        images_a = self.preprocess(images_a)
        images_b = self.preprocess(images_b)
        mu_a, mu_b = self.calc_mu(images_a), self.calc_mu(images_b)
        mu_aa, mu_bb, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
        sigma_aa = self.calc_mu(images_a**2) - mu_aa
        sigma_bb = self.calc_mu(images_b**2) - mu_bb
        sigma_ab = self.calc_mu(images_a * images_b) - mu_ab
        A1 = 2.0 * mu_ab + self.C1
        A2 = 2.0 * sigma_ab + self.C2
        B1 = (mu_aa + mu_bb + self.C1)
        B2 = (sigma_aa + sigma_bb + self.C2)
        mssim = (A1 * A2) / (B1 * B2)
        mssim = einops.reduce(mssim, 'N C H W->N', 'mean')
        return mssim

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        N, C, H, W = images.shape
        f = max(1, round(min(H, W) / 256))

        if f > 1 and self.downsample == True:
            images = F.avg_pool2d(images, kernel_size=f)

        return images

    def calc_mu(self, images):
        N, C, H, W = images.shape
        images = einops.rearrange(images, 'N C H W->(N C) 1 H W')
        mu = self.gauss_filter(images)
        mu = einops.rearrange(mu, '(N C) 1 H W->N C H W', N=N, C=C)
        mu = mu.to(torch.float64)
        return mu
