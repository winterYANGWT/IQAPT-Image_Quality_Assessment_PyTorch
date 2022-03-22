from .utils import check_img_size
import torch

__all__ = ['rgb_to_gray']


def base_colorspace_transformation(input_tensor, func):
    img_size = input_tensor.size()

    if len(img_size) == 4:
        N, C, H, W = img_size
        trgt_list = [None] * N

        for i in range(N):
            trgt_list[i] = func(input_tensor[i])

        return torch.stack(trgt_list)
    elif len(img_size) == 3:
        return func(input_tensor)
    elif len(img_size) == 2:
        return func(input_tensor.unsqueeze(0))
    else:
        msg = 'the size of input_tensor{} should be N,C,H,W or C,H,W or H,W.'.format(
            tuple(input_tensor.shape))
        raise ValueError(msg)


# def rgb_to_ycbcr(input_tensor):
#     def _rgb_to_ycbcr(img):
#         r,g,b=img[0],img[1],img[2]
#         y=0.299*r+0.587*g+0.114*b
#         cb=128-0.172*r-0.399*g+0.511*b
#         cr=128+0.511*r-0.428*g-0.083*b
#         return torch.stack([y,cb,cr])
#
#     return base_colorspace_transformation(input_tensor, _rgb_to_ycbcr)
#
#
# def ycbcr_to_rgb(input_tensor):
#     def _ycbcr_to_rgb(img):
#         y,cb,cr=img[0],img[1],img[2]
#         b=(cb-128)
#
#     return base_colorspace_transformation(input_tensor, _ycbcr_to_rgb)


def rgb_to_gray(input_tensor):
    def _rgb_to_gray(img):
        r, g, b = img[0], img[1], img[2]
        g = 0.299 * r + 0.587 * g + 0.114 * b
        return torch.stack([g])

    return base_colorspace_transformation(input_tensor, _rgb_to_gray)
