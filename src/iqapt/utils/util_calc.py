import torch
from .util_device import get_device, to_device

__all__ = ['sqrtm_newton_schulz']


def sqrtm_error(matrix, sqrt_matrix):
    norm_matrix = matrix.norm(p='fro')
    error = matrix - sqrt_matrix.mm(sqrt_matrix)
    norm_error = error.norm(p='fro')
    error = norm_error / norm_matrix
    return error


def sqrtm_newton_schulz(matrix, num_iters=100):
    '''
    based on https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    '''
    device = get_device(matrix)
    dtype=matrix.dtype
    dim = matrix.shape[0]
    norm = matrix.norm(p='fro')
    Y = matrix.div(norm)
    I = to_device(torch.eye(dim, dim, dtype=dtype), device)
    Z = to_device(torch.eye(dim, dim, dtype=dtype), device)
    zero_tensor = to_device(torch.zeros((1, ), dtype=dtype), device)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        sqrt_matrix = Y * torch.sqrt(norm)
        error = sqrtm_error(matrix, sqrt_matrix)

        if torch.isclose(error, zero_tensor, atol=1e-5):
            break

    return sqrt_matrix, error
