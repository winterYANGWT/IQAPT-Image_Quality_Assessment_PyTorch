from .metric import SingleImageMetric, torch
from .mse import MSE
from .. import utils

__all__ = ['PSNR']


class PSNR(SingleImageMetric):
    def __init__(self,
                 peak=1.0,
                 convert_to_gray=False,
                 soft_factor=1e-32) -> None:
        super().__init__()
        self.peak = peak
        self.convert_to_gray = convert_to_gray
        self.mse = MSE()
        self.soft_factor = soft_factor

    def calc(self, images_a: torch.Tensor,
             images_b: torch.Tensor) -> torch.Tensor:
        N, C, H, W = images_a.shape

        if self.convert_to_gray == True:
            if C == 1:
                mse = self.mse(images_a, images_b)
            elif C == 3:
                images_a_y = utils.image_colorspace.rgb_to_gray(images_a)
                images_b_y = utils.image_colorspace.rgb_to_gray(images_b)
                mse = self.mse(images_a_y, images_b_y)
            else:
                msg = f'the channel of img({C}) should be 1(Gray) or 3(RGB).'
                raise ValueError(msg)
        else:
            mse = self.mse(images_a, images_b)

        return 10 * torch.log10(self.peak**2 / (mse + self.soft_factor))
