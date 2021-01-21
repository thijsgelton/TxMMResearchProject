import os

import numpy as np
import torch
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, LoadInitState
from torch.nn.modules.loss import MSELoss
from torchvision import transforms

from ConvAutoEncoder import SegNet
from Datasets import AutoEncoderImageDataset

SIZE = (224, 224)
TRAIN_DIR = os.path.join(os.path.abspath("."), "..", "cropped_data")
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
    mean = np.array([0.5020, 0.4690, 0.4199])
    std = np.array([0.2052, 0.2005, 0.1966])
    aug_tran = transforms.Compose([
        transforms.Resize(SIZE, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = AutoEncoderImageDataset(TRAIN_DIR, transform=aug_tran)

    net.fit(dataset, y=None)
