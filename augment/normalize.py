from torchvision import transforms


class Normalize:
    def __init__(self, dataset="imagenet"):
        if dataset == "none":
            self.transform = lambda x: x
        else:
            self.transform = transforms.Normalize(*get_mean_and_std(dataset))

    def __call__(self, img):
        return self.transform(img)
    

def get_mean_and_std(dataset):
    if dataset == "cifar10":
        means = [0.49139968, 0.48215841, 0.44653091]
        stds = [0.24703223, 0.24348513, 0.26158784]
    elif dataset == "cifar100":
        means = [n / 255. for n in [129.3, 124.1, 112.4]]
        stds = [n / 255. for n in [68.2, 65.4, 70.4]]
    elif dataset == "svhn":
        means = [0.4376821, 0.4437697, 0.47280442]
        stds = [0.19803012, 0.20101562, 0.19703614]
    elif dataset == "imagenet":
        # imagenet statistics
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return means, stds
