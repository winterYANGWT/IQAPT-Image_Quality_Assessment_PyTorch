from torch.hub import load_state_dict_from_url
import torch.nn as nn

channels = {
    'alex': [64, 192, 384, 256, 256],
    'vgg': [64, 128, 256, 512, 512],
    'squeeze': [64, 128, 256, 384, 384, 512, 512]
}
models_url_dict = {
    'alex':
    'https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/alex.pth?raw=true',
    'vgg':
    'https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth?raw=true',
    'squeeze':
    'https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/squeeze.pth?raw=true'
}


class LinearCalibrationLayer(nn.Module):
    def __init__(self, net='vgg') -> None:
        super().__init__()
        self.channels = channels[net]
        self.num_layers = len(self.channels)
        lins = []

        for i, channel in enumerate(self.channels):
            lin = LinearLayer(channel)
            self.add_module(f'lin{i}', lin)
            lins.append(lin)

        self.lins = nn.ModuleList(lins)
        self.load_state_dict(load_state_dict_from_url(models_url_dict[net],
                                                      map_location='cpu',
                                                      progress=True),
                             strict=False)

    def forward(self, input_tensor_list):
        output_tensor_list = []

        for i, input_tensor in enumerate(input_tensor_list):
            output_tensor = self.lins[i](input_tensor)
            output_tensor_list.append(output_tensor)

        return output_tensor_list


class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        layers = [nn.Identity()]
        layers += [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.model(input_tensor)
