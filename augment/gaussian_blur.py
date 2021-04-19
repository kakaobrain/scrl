import random

from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np


class GaussianBlur(object):
    """Blur a single image on CPU.
    """

    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        radius = kernel_size // 2
        kernel_size = radius * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.k = kernel_size
        self.r = radius

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radius),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class ResizeBlur(object):
    """Cost efficient alternative of Gaussian blur.
    """
    def __init__(self, input_size, max_level=3, interpolation=Image.BICUBIC):
        self.input_size = input_size
        self.max_level = max_level
        self.factors = [1.1, 1.2, 1.5, 2.0, 4.0, 8.0]
        self.interpolation = interpolation

    def __call__(self, img):
        level = np.random.randint(0, self.max_level)
        w, h = img.size
        dn_size = (int(h // self.factors[level]), 
                   int(w // self.factors[level]))
        up_size = (h, w)
        # note that interpolation method is different from the reference code (AREA)
        img = transforms.functional.resize(img, dn_size, interpolation=self.interpolation)
        img = transforms.functional.resize(img, up_size, interpolation=self.interpolation)
        return img
