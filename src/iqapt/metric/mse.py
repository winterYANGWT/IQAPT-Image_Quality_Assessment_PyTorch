from .metric import SingleImageMetric, torch
import einops

__all__ = ['MSE']


class MSE(SingleImageMetric):
    def __init__(self) -> None:
        super().__init__()

    def calc(self, images_a: torch.Tensor,
             images_b: torch.Tensor) -> torch.Tensor:
        N, C, H, W = images_a.shape
        diff = (images_a - images_b)**2
        mse=einops.reduce(diff,'N C H W->N','mean')
        return mse
