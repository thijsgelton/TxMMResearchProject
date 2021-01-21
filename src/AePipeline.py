import os

import torch
from sklearn.pipeline import Pipeline
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, LoadInitState, LRScheduler
from torch import cuda, device
from torch.backends import cudnn
from torch.nn.modules.loss import BCEWithLogitsLoss, BCELoss, MSELoss
from torch.optim.lr_scheduler import CyclicLR
from torchsummary import summary
from torchvision import transforms
from AutoEncoder import ConvAutoEncoder, AutoEncoder
from ConvAutoEncoder import SegNet
from Datasets import AutoEncoderImageDataset
import matplotlib.pyplot as plt
import numpy as np

# Basic Transforms
from utils import UnNormalize

SIZE = (224, 224)  # Resize the image to this shape
TRAIN_DIR = os.path.join(os.path.abspath("."), "..", "cropped_data")
#
cp = Checkpoint(dirname='segnet_mse_no_sigmoid_sgd_150ep_b8_lr_0.01_30enc/checkpoints')
train_end_cp = TrainEndCheckpoint(dirname='segnet_mse_no_sigmoid_sgd_150ep_b8_lr_0.01_30enc/checkpoints')
load_state = LoadInitState(checkpoint=cp)
net = NeuralNetRegressor(
    SegNet,
    module__encoding_size=30,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    max_epochs=150,
    batch_size=8,
    criterion=MSELoss,
    lr=0.01,
    iterator_train__shuffle=True,
    optimizer=torch.optim.SGD,
    optimizer__momentum=.9,
    callbacks=[cp, train_end_cp, load_state]
)

if __name__ == '__main__':
    # net.initialize()
    # net.load_params(checkpoint=cp)
    # summary(net.module_, input_size=(3, 224, 224))

    # Simple Data Augmentation
    mean = np.array([0.5020, 0.4690, 0.4199])
    std = np.array([0.2052, 0.2005, 0.1966])
    aug_tran = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.Resize(SIZE, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create Dataset
    dataset = AutoEncoderImageDataset(TRAIN_DIR, transform=aug_tran)

    net.fit(dataset, y=None)
