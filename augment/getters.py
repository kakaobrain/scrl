from collections.abc import Iterable

import augment.transforms as transforms
from utils import Config
from PIL import Image

INPUT_SIZE = 224
PAD_SIZE = 32
RESIZE_METHOD = Image.BICUBIC
BLUR_METHOD = ('gaussian', 'resized')[1]


class MultiViewDataInjector(object):
    def __init__(self, transforms):
        if not isinstance(transforms, Iterable):
            transforms = (transforms,)
        self.transforms = transforms

    def __call__(self, sample):
        return [transform(sample) for transform in self.transforms]


def get_transforms(cfg, train):
    if train:
        return get_train_transforms(cfg)
    else:
        return get_test_transforms(cfg)


def get_train_transforms(cfg: Config):
    aug = cfg.augment
    with_trans_info = cfg.network.scrl.enabled is True

    if aug.type == "none":
        transforms_ = [
            get_center_crop_transforms(input_size=aug.input_size,
                                       with_trans_info=with_trans_info)
        ] * 2
    elif aug.type == "simple":
        transforms_ = [
            get_simple_transforms(input_size=aug.input_size,
                                crop_scale=aug.crop_scale,
                                with_trans_info=with_trans_info)
        ] * 2
    elif aug.type == "simclr":
        transforms_ = [
            get_simclr_transforms_for_view_1(input_size=aug.input_size,
                                             crop_scale=aug.crop_scale,
                                             with_trans_info=with_trans_info),
            get_simclr_transforms_for_view_2(input_size=aug.input_size,
                                             crop_scale=aug.crop_scale,
                                             with_trans_info=with_trans_info),
        ]
    else:
        raise ValueError(f"Unexpected aug_type: {aug.type}")

    return MultiViewDataInjector(transforms=transforms_)


def get_test_transforms(cfg: Config):
    transforms_ = get_center_crop_transforms(input_size=cfg.augment.input_size,
                                             with_trans_info=False)
    return MultiViewDataInjector(transforms=[transforms_])


def get_center_crop_transforms(input_size=INPUT_SIZE, pad_size=PAD_SIZE, with_trans_info=False):
    resize_ratio = (input_size + pad_size) / input_size
    return transforms.Compose([
        transforms.Resize(round(input_size * resize_ratio), interpolation=RESIZE_METHOD),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(),
    ], with_trans_info=with_trans_info)


def get_simple_transforms(input_size=INPUT_SIZE, crop_scale=(0.08, 1.0), with_trans_info=False):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=input_size,
                                     scale=crop_scale,
                                     interpolation=RESIZE_METHOD),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(),
    ], with_trans_info=with_trans_info)


def _get_blur_method(input_size):
    if BLUR_METHOD == 'gaussian':
        blur_method = transforms.GaussianBlur(kernel_size=int(0.1 * input_size))
    elif BLUR_METHOD == 'resized':
        blur_method = transforms.ResizeBlur(input_size=input_size,
                                            max_level=3,
                                            interpolation=RESIZE_METHOD)
    else:
        Exception(f"Unknown blur method: {BLUR_METHOD}")
    return blur_method


def get_simclr_transforms_in_byol(p_gaussian_blur, p_solarize, input_size=INPUT_SIZE,
                                  crop_scale=(0.08, 1.0), p_hflip=0.5, with_trans_info=False):
    # get a set of data augmentation transformations as described in BYOL paper(Table 6).
    # https://github.com/deepmind/deepmind-research/blob/85187de3dc84ebbde0605cb55ac89e4419c87992/byol/utils/augmentations.py
    return transforms.Compose([
        transforms.RandomResizedCrop(size=input_size,
                                     scale=crop_scale,
                                     interpolation=RESIZE_METHOD),
        transforms.RandomHorizontalFlip(p=p_hflip),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([_get_blur_method(input_size)], p=p_gaussian_blur),
        transforms.RandomApply([transforms.Solarize(threshold=0.5)],
                               p=p_solarize),
        transforms.ToTensor(),
        transforms.Normalize(),
    ], with_trans_info=with_trans_info)


def get_simclr_transforms_for_view_1(input_size=INPUT_SIZE,
                                     crop_scale=(0.08, 1.0), with_trans_info=False):
    return get_simclr_transforms_in_byol(
        p_gaussian_blur=1.0,
        p_solarize=0.0,
        p_hflip=0.5,
        input_size=input_size,
        crop_scale=crop_scale,
        with_trans_info=with_trans_info,
    )


def get_simclr_transforms_for_view_2(input_size=INPUT_SIZE, p_hflip=0.5,
                                     crop_scale=(0.08, 1.0), with_trans_info=False):
    return get_simclr_transforms_in_byol(
        p_gaussian_blur=0.1,
        p_solarize=0.2,
        p_hflip=0.5,
        input_size=input_size,
        crop_scale=crop_scale,
        with_trans_info=with_trans_info,
    )
