from typing import NamedTuple, List, Tuple
from functools import wraps

import torch
import torch.nn.functional as F
import random
from torchvision.transforms import (Resize, CenterCrop, RandomHorizontalFlip,
                                    ColorJitter, RandomGrayscale, ToTensor)
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import ImageOps

from augment.gaussian_blur import GaussianBlur, ResizeBlur
from augment.normalize import Normalize


class ImageWithTransInfo(NamedTuple):
    """to improve readability"""
    image: torch.Tensor  # image
    transf: List         # cropping coord. in the original image + flipped or not
    ratio: List          # resizing ratio w.r.t. the original image
    size: List           # size (width, height) of the original image


def free_pass_trans_info(func):
    """Wrapper to make the function bypass the second argument(transf)."""
    @wraps(func)
    def decorator(img, transf, ratio):
        return func(img), transf, ratio
    return decorator


def _with_trans_info(transform):
    """use with_trans_info function if possible, or wrap original __call__."""
    if hasattr(transform, 'with_trans_info'):
        transform = transform.with_trans_info
    else:
        transform = free_pass_trans_info(transform)
    return transform


def _get_size(size):
    if isinstance(size, int):
        oh, ow = size, size
    else:
        oh, ow = size
    return oh, ow


def _update_transf_and_ratio(transf_global, ratio_global,
                             transf_local=None, ratio_local=None):
    if transf_local:
        i_global, j_global, *_ = transf_global
        i_local, j_local, h_local, w_local = transf_local
        i = int(round(i_local / ratio_global[0] + i_global))
        j = int(round(j_local / ratio_global[1] + j_global))
        h = int(round(h_local / ratio_global[0]))
        w = int(round(w_local / ratio_global[1]))
        transf_global = [i, j, h, w]

    if ratio_local:
        ratio_global = [g * l for g, l in zip(ratio_global, ratio_local)]

    return transf_global, ratio_global


class Compose(object):
    def __init__(self, transforms, with_trans_info=False, seed=None):
        self.transforms = transforms
        self.with_trans_info = with_trans_info
        self.seed = seed
        
    @property
    def with_trans_info(self):
        return self._with_trans_info
    
    @with_trans_info.setter
    def with_trans_info(self, value):
        self._with_trans_info = value

    def __call__(self, *args, **kwargs):
        if self.with_trans_info:
            return self._call_with_trans_info(*args, **kwargs)
        return self._call_default(*args, **kwargs)

    def _call_default(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def _call_with_trans_info(self, img):
        w, h = img.size
        transf = [0, 0, h, w]
        ratio = [1., 1.]

        for t in self.transforms:
            t = _with_trans_info(t)
            try:
                if self.seed:
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
                img, transf, ratio = t(img, transf, ratio)
            except Exception as e:
                raise Exception(f'{e}: from {t.__self__}')

        return ImageWithTransInfo(img, transf, ratio, (h, w))


class CenterCrop(transforms.CenterCrop):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size
        oh, ow = _get_size(self.size)
        i = int(round((w - ow) * 0.5))
        j = int(round((h - oh) * 0.5))
        transf_local = [i, j, oh, ow]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, None)
        return F.center_crop(img, self.size), transf, ratio


class Resize(transforms.Resize):
    def with_trans_info(self, img, transf, ratio):
        w, h = img.size  # PIL.Image
        resized_img = F.resize(img, self.size, self.interpolation)
        # get the size directly from resized image rather than using _get_size()
        # since only smaller edge of the image will be matched in this class.
        ow, oh = resized_img.size
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, None, ratio_local)
        return resized_img, transf, ratio
    
    
class RandomResizedCrop(transforms.RandomResizedCrop):
    def with_trans_info(self, img, transf, ratio):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        
        oh, ow = _get_size(self.size)
        transf_local = [i, j, h, w]
        ratio_local = [oh / h, ow / w]
        transf, ratio = _update_transf_and_ratio(
            transf, ratio, transf_local, ratio_local)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, transf, ratio


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def with_trans_info(self, img, transf, ratio):
        if torch.rand(1) < self.p:
            transf.append(True)
            return F.hflip(img), transf, ratio
        transf.append(False)
        return img, transf, ratio


class RandomOrder(transforms.RandomOrder):
    def with_trans_info(self, img, transf, ratio):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            t = _with_trans_info(self.transforms[i])
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio


class RandomApply(transforms.RandomApply):
    def with_trans_info(self, img, transf, ratio):
        if self.p < random.random():
            return img, transf, ratio
        for t in self.transforms:
            t = _with_trans_info(t)
            img, transf, ratio = t(img, transf, ratio)
        return img, transf, ratio


class Solarize(object):
    def __init__(self, threshold):
        assert 0 < threshold < 1
        self.threshold = round(threshold * 256)

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)

    def __repr__(self):
        attrs = f"(min_scale={self.threshold}"
        return self.__class__.__name__ + attrs
