import torch.nn.functional as F
from .metric import SingleImageMetric, torch
from . import utils
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
        NP = window_size**2
        self.conv_norm = NP / (NP - 1)
        self.C1 = (K1 * L)**2
        self.C2 = (K2 * L)**2

    def calc(self, images_a: torch.Tensor,
             images_b: torch.Tensor) -> torch.Tensor:
        N, C, H, W = images_a.shape
        f = max(1, round(min(H, W) / 256))

        if f > 1 and self.downsample == True:
            images_a = F.avg_pool2d(images_a, kernel_size=f)
            images_b = F.avg_pool2d(images_b, kernel_size=f)

        images_a = einops.rearrange(images_a, 'N C H W->(N C) 1 H W')
        images_b = einops.rearrange(images_b, 'N C H W->(N C) 1 H W')
        mu_a = self.gauss_filter(images_a)
        mu_b = self.gauss_filter(images_b)
        mu_aa, mu_bb, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
        sigma_aa = self.gauss_filter(images_a**2) - mu_aa
        sigma_bb = self.gauss_filter(images_b**2) - mu_bb
        sigma_ab = self.gauss_filter(images_a * images_b) - mu_ab
        A1 = 2.0 * mu_ab + self.C1
        A2 = 2.0 * sigma_ab + self.C2
        B1 = (mu_aa + mu_bb + self.C1)
        B2 = (sigma_aa + sigma_bb + self.C2)
        mssim = (A1 * A2) / (B1 * B2)
        mssim = einops.rearrange(mssim, '(N C) 1 H W->N C H W', N=N, C=C)
        mssim = torch.mean(mssim, dim=[1, 2, 3])
        return mssim
