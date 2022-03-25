import os
import os.path as path
from prettytable import PrettyTable
import functools
import torchvision as tv
import iqapt
from PIL import Image
from skimage import metrics
import numpy as np
import piq
from file_dict import *
import lpips

CUDA=True

os.chdir(path.dirname(path.abspath(__file__)))
open_images_from_dir = lambda dir: [
    Image.open(os.path.join(dir, file)).convert('RGB')
    for file in os.listdir(dir)
]

open_images = lambda d: {
    name: Image.open(file).convert('RGB')
    for (name, file) in d.items()
}

dict_map = lambda d, func: {name: func(value) for (name, value) in d.items()}
list_map=lambda l,func:[func(value) for value in l]

# transfer func
t_image2tensor = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Lambda(lambda img:img.cuda())]) if CUDA==True else tv.transforms.ToTensor()
image2tensor = lambda image: t_image2tensor(image).unsqueeze(0)
image2np = lambda image: np.array(image) / 255.0


def test_MSE():
    print('MSE: ')
    iqapt_mse = iqapt.MSE()
    table = PrettyTable(['ref_image', 'test_image', 'iqapt', 'scikit'])

    if CUDA==True:
        iqapt_mse=iqapt_mse.cuda()

    for p_images in MSE_IMAGES:
        images = open_images(p_images)
        tensor_images = dict_map(images, image2tensor)
        np_images = dict_map(images, image2np)
        iqapt_result = iqapt_mse(tensor_images['ref'],
                                 tensor_images['test']).item()
        scikit_result = metrics.mean_squared_error(np_images['ref'],
                                                   np_images['test'])
        table.add_row([
            path.split(p_images['ref'])[-1],
            path.split(p_images['test'])[-1], iqapt_result, scikit_result
        ])

    print(table)


def test_PSNR():
    print('PSNR: ')
    iqapt_psnr = iqapt.PSNR()
    piq_psnr = piq.psnr
    iqapt_psnr_g = iqapt.PSNR(convert_to_gray=True)
    piq_psnr_g = functools.partial(piq.psnr, convert_to_greyscale=True)
    table = PrettyTable([
        'ref_image', 'test_image', 'iqapt_gray', 'piq_gray', 'iqapt', 'piq',
        'scikit'
    ])

    if CUDA==True:
        iqapt_psnr=iqapt_psnr.cuda()
        iqapt_psnr_g=iqapt_psnr_g.cuda()

    for p_images in PSNR_IMAGES:
        images = open_images(p_images)
        tensor_images = dict_map(images, image2tensor)
        np_images = dict_map(images, image2np)
        iqapt_result_g = iqapt_psnr_g(tensor_images['ref'],
                                      tensor_images['test']).item()
        piq_result_g = piq_psnr_g(tensor_images['ref'],
                                  tensor_images['test']).item()
        iqapt_result = iqapt_psnr(tensor_images['ref'],
                                  tensor_images['test']).item()
        piq_result = piq_psnr(tensor_images['ref'],
                              tensor_images['test']).item()
        scikit_result = metrics.peak_signal_noise_ratio(
            np_images['ref'], np_images['test'])
        table.add_row([
            path.split(p_images['ref'])[-1],
            path.split(p_images['test'])[-1], iqapt_result_g, piq_result_g,
            iqapt_result, piq_result, scikit_result
        ])

    print(table)


def test_SSIM():
    print('SSIM: ')
    iqapt_ssim = iqapt.MSSIM(window_size=11)
    piq_ssim = piq.ssim
    table = PrettyTable(['ref_image', 'test_image', 'iqapt', 'piq', 'scikit'])

    if CUDA==True:
        iqapt_ssim=iqapt_ssim.cuda()

    for p_images in SSIM_IMAGES:
        images = open_images(p_images)
        tensor_images = dict_map(images, image2tensor)
        np_images = dict_map(images, image2np)
        iqapt_result = iqapt_ssim(tensor_images['ref'],
                                  tensor_images['test']).item()
        piq_result = piq_ssim(tensor_images['ref'],
                              tensor_images['test']).item()
        scikit_result = metrics.structural_similarity(np_images['ref'],
                                                      np_images['test'],
                                                      channel_axis=2,
                                                      win_size=11,
                                                      data_range=1,
                                                      gaussian_weights=True)

        table.add_row([
            path.split(p_images['ref'])[-1],
            path.split(p_images['test'])[-1], iqapt_result, piq_result,
            scikit_result
        ])

    print(table)


def test_LPIPS():
    print('LPIPS: ')
    iqapt_lpips = iqapt.LPIPS()
    piq_lpips = piq.LPIPS()
    lpips_lpips = lpips.LPIPS(net='vgg')
    table = PrettyTable(
        ['ref_image', 'test_image', 'iqapt', 'piq', 'lpips(official code)'])

    if CUDA==True:
        iqapt_lpips=iqapt_lpips.cuda()
        lpips_lpips=lpips_lpips.cuda()

    for p_images in LPIPS_IMAGES:
        images = open_images(p_images)
        tensor_images = dict_map(images, image2tensor)
        iqapt_result = iqapt_lpips(tensor_images['ref'],
                                   tensor_images['test']).item()
        piq_result = piq_lpips(tensor_images['ref'],
                               tensor_images['test']).item()
        lpips_result = lpips_lpips(tensor_images['ref'],
                                   tensor_images['test'],
                                   normalize=True).item()

        table.add_row([
            path.split(path.split(p_images['ref'])[-1])[-1],
            path.split(p_images['test'])[-1], iqapt_result, piq_result,
            lpips_result
        ])

    print(table)


def test_FID():
    print('FID: ')
    iqapt_fid = iqapt.FID()
    piq_fid = piq.FID()
    table = PrettyTable(['ref_dir', 'test_dir', 'iqapt', 'piq'])

    if CUDA==True:
        iqapt_fid=iqapt_fid.cuda()

    src_images = open_images_from_dir('./Image/FID/source_images')
    trgt_images = open_images_from_dir('./Image/FID/generated_images')

    src_images = list_map(src_images,t_image2tensor)
    trgt_images = list_map(trgt_images,t_image2tensor)

    iqapt_result, features_a, features_b = iqapt_fid(src_images, trgt_images)
    iqapt_result = iqapt_result.item()
    piq_result = piq_fid.compute_metric(features_a, features_b).item()
    table.add_row(
        ['source_images', 'generated_images', iqapt_result, piq_result])
    print(table)


if __name__ == '__main__':
    test_MSE()
    test_PSNR()
    test_SSIM()
    test_LPIPS()
    test_FID()
