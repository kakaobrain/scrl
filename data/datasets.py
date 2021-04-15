import logging
import random

import torch
from torchvision import datasets

log = logging.getLogger('main')


def get_dataset(data_name: str, data_root: str, train: bool, 
                transform, num_subsample=-1):
    # dataset
    if data_name == 'imagenet':
        dataset_cls = get_dataset_cls(data_name)
        split = 'train' if train else 'val'
        dataset = dataset_cls(root=data_root, split=split, transform=transform)
        num_classes = len(dataset.classes)
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")

    log.info(f"[Dataset] {data_name}(train={'True' if train else 'False'}) / "
             f"{dataset.__len__()} images are available.")

    # use smaller subset when debugging
    if num_subsample > 0:
        log.info(f"[Dataset] sample {num_subsample} images from the dataset in "
                 f"{'train' if train else 'test'} set.")
        num_subsample = min(num_subsample, len(dataset))
        indices = random.choices(range(len(dataset)), k=num_subsample)
        dataset = torch.utils.data.Subset(dataset, list(indices))
        
    return dataset, num_classes


def get_dataset_cls(name):
    try:
        return {
            'imagenet': datasets.ImageNet,
            # add your custom dataset class here
        }[name]
    except KeyError:
        raise KeyError(f"Unexpected dataset name: {name}")
