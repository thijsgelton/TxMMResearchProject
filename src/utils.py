import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms

from Datasets import AutoEncoderImageDataset


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def compute_mean_and_std():
    TRAIN_DIR = os.path.join(os.path.abspath("."), "..", "data")
    dataset = AutoEncoderImageDataset(TRAIN_DIR,
                                      transform=transforms.Compose(
                                          [transforms.Resize((224, 224), interpolation=3), ToTensor()]))
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples

    temp = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        elementNum = data.size(0) * data.size(2) * data.size(3)
        data = data.permute(1, 0, 2, 3).reshape(3, elementNum)
        temp += ((data - mean.repeat(elementNum, 1).permute(1, 0)) ** 2).sum(1) / (elementNum * batch_samples)
        nb_samples += batch_samples

    std = torch.sqrt(temp / nb_samples)
    print(mean)
    print(std)


if __name__ == "__main__":
    compute_mean_and_std()
