from .alex_net import AlexNet
from .vgg16 import Vgg16
from .squeeze_net import SqueezeNet
from .scaling_layer import ScalingLayer
from .linear_layer import LinearCalibrationLayer

feature_extractors = {'alex': AlexNet, 'vgg': Vgg16, 'squeeze': SqueezeNet}

__all__ = ['AlexNet', 'Vgg16', 'SqueezeNet']
