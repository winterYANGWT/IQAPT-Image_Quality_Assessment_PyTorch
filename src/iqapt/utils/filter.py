from abc import ABC, abstractclassmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Filter(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, input_tensor):
        input_size = input_tensor.size()

        if len(input_size) == 4:
            N, C, H, W = input_size
            results = self.filter(input_tensor)
        elif len(input_size) == 3:
            C, H, W = input_size
            results = self.filter(input_tensor.unsqueeze(0))
        elif len(input_size) == 2:
            H, W = input_size
            results = self.filter(input_tensor.unsqueeze(0).unsqueeze(0))
        else:
            msg = 'the shape of input_tensor{} should be N,C,H,W or C,H,W or H,W.'.format(
                tuple(input_tensor.shape))
            raise ValueError(msg)

        return results

    @abstractclassmethod
    def filter(self, img):
        raise NotImplementedError('filter should be implemented by subclass.')


class GaussianFilter(Filter):
    def __init__(self, kernel_size, sigma) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_weight = self.generate_weight(self.kernel_size,
                                                    self.sigma)

    def generate_weight(self, kernel_size, sigma):
        one_D_weight = torch.Tensor([
            math.exp(-(x - kernel_size // 2)**2 / float(2 * sigma**2))
            for x in range(kernel_size)
        ])
        one_D_weight = one_D_weight / one_D_weight.sum()
        _one_D_weight = one_D_weight.unsqueeze(1)
        two_D_weight = _one_D_weight.mm(_one_D_weight.t()).float()
        two_D_weight = two_D_weight.unsqueeze(0).unsqueeze(0)
        two_D_weight = nn.Parameter(two_D_weight)
        return two_D_weight

    def filter(self, img):
        return F.conv2d(img, weight=self.gaussian_weight)
