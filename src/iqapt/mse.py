from re import A
from .metric import SingleImageMetric, torch

__all__ = ['MSE']


class MSE(SingleImageMetric):
    def __init__(self) -> None:
        super().__init__()

    def calc(self, images_a: torch.Tensor,
             images_b: torch.Tensor) -> torch.Tensor:
        N, C, H, W = images_a.shape
        diff = (images_a - images_b)**2 / (C * H * W)
        mse = torch.einsum('NCHW->N', diff)
        return mse
