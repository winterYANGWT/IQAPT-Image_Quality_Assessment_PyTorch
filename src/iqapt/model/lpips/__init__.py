from .alex_net import AlexNet
from .vgg import Vgg
from .squeeze_net import SqueezeNet
from .scaling_layer import ScalingLayer
from .linear_layer import LinearCalibrationLayer

feature_extractors = {'alex': AlexNet, 'vgg': Vgg, 'squeeze': SqueezeNet}
