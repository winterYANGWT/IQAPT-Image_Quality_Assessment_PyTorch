import torch
import torch.nn as nn
from typing import List, Union
import math


class SingleImageMetric(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, images_a: torch.Tensor, images_b: torch.Tensor):
        a_shape = images_a.shape
        b_shape = images_b.shape

        if a_shape != b_shape:
            msg = f'images_a and images_b should have the same shape, but got images_a({a_shape}) and images_b({b_shape}).'
            raise ValueError(msg)

        if len(a_shape) != 4:
            msg = 'the shape of images_a and images_b should be (N, C, H, W).'
            raise ValueError(msg)

        images_a=images_a.to(torch.float64)
        images_b=images_b.to(torch.float64)
        results = self.calc(images_a, images_b)
        return results

    def calc(self, images_a: torch.Tensor,
             images_b: torch.Tensor) -> torch.Tensor:
        '''
        implemented by sub class.
        '''
        msg = 'calc should be implemented by subclass.'
        raise NotImplementedError(msg)



class MultiImageMetric(nn.Module):
    def __init__(self, num_batch=1):
        super().__init__()
        self.num_batch = num_batch

    @torch.no_grad()
    def forward(self, images_a: Union[torch.Tensor, List[torch.Tensor]],
                images_b: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(images_a, torch.Tensor):
            features_a = self.__process_tensor(images_a, 'images_a')
        elif isinstance(images_a, list):
            features_a = self.__process_list(images_a, 'images_a')
        else:
            msg = 'images_a should be torch.Tensor or list.'
            raise ValueError(msg)

        if isinstance(images_b, torch.Tensor):
            features_b = self.__process_tensor(images_b, 'images_b')
        elif isinstance(images_b, list):
            features_b = self.__process_list(images_b, 'images_b')
        else:
            msg = 'images_b should be torch.Tensor or list.'
            raise ValueError(msg)

        result = self.calc(features_a, features_b)
        return result, features_a, features_b

    def __process_tensor(self, images, name):
        shape = images.shape
        N = shape[0]
        features = []

        if len(shape) != 4 or len(shape) != 4:
            msg = f'the shape of {name} should be (N, C, H, W).'
            raise ValueError(msg)

        images=images.to(torch.float64)

        for i in range(math.ceil(N / self.num_batch)):
            start_index = max(0, i * self.num_batch)
            end_index = min((i + 1) * self.num_batch, N)
            batch_images_a = images[start_index:end_index]
            features.append(self.calc_features(batch_images_a))

        features = torch.cat(features, dim=0)
        return features

    def __process_list(self, images, name):
        features = []

        for image in images:
            if len(image.shape) != 3:
                msg = f'the image in list {name} should be (C,H,W).'
                raise ValueError(msg)

        for image in images:
            image=image.to(torch.float64)
            feature = self.calc_features(image.unsqueeze(0))
            features.append(feature)

        features = torch.cat(features, dim=0)
        return features

    def calc(self, features_a: torch.Tensor,
             features_b: torch.Tensor) -> torch.Tensor:
        '''
        Implemented by sub class.
        '''
        msg = 'calc should be implemented by subclass.'
        raise NotImplementedError(msg)

    def calc_features(self, images: torch.Tensor) -> torch.Tensor:
        return images
