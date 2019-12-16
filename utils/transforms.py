import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms.functional as tf


def custom_greyscale_to_tensor(include_rgb):
    def _inner(img):
        grey_img_tensor = tf.to_tensor(tf.to_grayscale(img, num_output_channels=1))
        result = grey_img_tensor  # 1, 96, 96 in [0, 1]
        assert (result.size(0) == 1)

        if include_rgb:  # greyscale last
            img_tensor = tf.to_tensor(img)
            result = torch.cat([img_tensor, grey_img_tensor], dim=0)
            assert (result.size(0) == 4)

        return result

    return _inner


def sobel_process(imgs, include_rgb, using_IR=False):
    bn, c, h, w = imgs.size()

    if not using_IR:
        if not include_rgb:
            assert (c == 1)
            grey_imgs = imgs
        else:
            assert (c == 4)
            grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
            rgb_imgs = imgs[:, :3, :, :]
    else:
        if not include_rgb:
            assert (c == 2)
            grey_imgs = imgs[:, 0, :, :].unsqueeze(1)  # underneath IR
            ir_imgs = imgs[:, 1, :, :].unsqueeze(1)
        else:
            assert (c == 5)
            rgb_imgs = imgs[:, :3, :, :]
            grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
            ir_imgs = imgs[:, 4, :, :].unsqueeze(1)

    sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(
        torch.Tensor(sobel1).cuda().float().unsqueeze(0).unsqueeze(0))
    dx = conv1(Variable(grey_imgs)).data

    sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(
        torch.from_numpy(sobel2).cuda().float().unsqueeze(0).unsqueeze(0))
    dy = conv2(Variable(grey_imgs)).data

    sobel_imgs = torch.cat([dx, dy], dim=1)
    assert (sobel_imgs.shape == (bn, 2, h, w))

    if not using_IR:
        if include_rgb:
            sobel_imgs = torch.cat([rgb_imgs, sobel_imgs], dim=1)
            assert (sobel_imgs.shape == (bn, 5, h, w))
    else:
        if include_rgb:
            # stick both rgb and ir back on in right order (sobel sandwiched inside)
            sobel_imgs = torch.cat([rgb_imgs, sobel_imgs, ir_imgs], dim=1)
        else:
            # stick ir back on in right order (on top of sobel)
            sobel_imgs = torch.cat([sobel_imgs, ir_imgs], dim=1)

    return sobel_imgs
