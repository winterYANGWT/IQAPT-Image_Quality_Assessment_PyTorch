import torch
import torch.nn.functional as F


def check_img_size(img_a, img_b):
    if img_a.size() != img_b.size():
        return None
    else:
        return img_a.size()


def unfold_img(img, window_size):
    if len(img.size()) == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(img.size()) == 3:
        img = img.unsqueeze(0)
    elif len(img.size()) == 4:
        img = img
    else:
        msg = 'the size of img{} should be N,C,H,W or C,H,W or H,W.'.format(
            tuple(img.shape))
        raise ValueError(msg)

    N, C, H, W = img.size()
    unfold = torch.nn.Unfold(window_size)
    unfold_img = unfold(img)
    unfold_img = unfold_img.permute(0, 2, 1)
    unfold_img = unfold_img.reshape(N, -1, C, window_size, window_size)
    return unfold_img


def normalize_tensor(input_tensor, eps=0.1**10):
    tensor_dims = len(input_tensor.size())

    if tensor_dims == 4:
        start_dim = 1
    elif tensor_dims == 3:
        start_dim = 0
    elif tensor_dims == 2:
        input_tensor = input_tensor.unsqueeze(0)
        start_dim = 0
    else:
        msg = 'the size of input_tensor{} should be N,C,H,W or C,H,W or H,W.'.format(
            tuple(input_tensor.shape))
        raise ValueError(msg)

    norm_factor = torch.sqrt(
        torch.sum(input_tensor**2, dim=start_dim, keepdim=True))
    return input_tensor / (norm_factor + eps)


def spatial_average(input_tensor, keepdim=True):
    tensor_dims = len(input_tensor.size())

    if tensor_dims == 4:
        start_dim = 2
    elif tensor_dims == 3:
        start_dim = 1
    elif tensor_dims == 2:
        start_dim = 0
    else:
        msg = 'the size of input_tensor{} should be N,C,H,W or C,H,W or H,W.'.format(
            tuple(input_tensor.shape))
        raise ValueError(msg)

    return torch.mean(input_tensor,
                      dim=list(range(start_dim, tensor_dims)),
                      keepdim=keepdim)


def upsample(input_tensor, out_size, mode='bilinear'):
    tensor_dims = len(input_tensor.size())

    if tensor_dims == 2:
        input_tensor = torch.unsqueeze(input_tensor, 0)
    elif tensor_dims < 2:
        msg = 'the size of input_tensor{} should be N,C,H,W or C,H,W or H,W.'.format(
            tuple(input_tensor.shape))
        raise ValueError(msg)

    return F.upsample(input_tensor,
                      size=out_size,
                      mode=mode,
                      align_corners=False)
