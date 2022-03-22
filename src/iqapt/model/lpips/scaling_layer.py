import torch
import torch.nn as nn


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            'shift',
            torch.tensor([-.030, -.088, -.188]).reshape(1, 3, 1, 1))
        self.register_buffer(
            'scale',
            torch.tensor([.458, .448, .450]).reshape(1, 3, 1, 1))

    def forward(self, input_tensor):
        input_tensor = 2.0 * input_tensor - 1.0
        return (input_tensor - self.shift) / self.scale
