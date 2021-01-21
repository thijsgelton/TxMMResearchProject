from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint, TrainEndCheckpoint
from torch import nn
from torch.nn import MSELoss
from torchsummary import summary
from torchvision import transforms

from Blocks import BasicConvBlock, UpConvBlock
from ConvAutoEncoder import SegNet
from utils import init_weights


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4 * 13 * 13, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 4 * 13 * 13),
            nn.Unflatten(1, (4, 13, 13)),
            nn.ConvTranspose2d(4, 4, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 4, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, coding_size, img_channels=3, hidden_dims=(16, 32, 16, 8, 2)):
        super(AutoEncoder, self).__init__()

        # encoder
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = BasicConvBlock(ch_in=img_channels, ch_out=hidden_dims[0])
        self.Conv2 = BasicConvBlock(ch_in=hidden_dims[0], ch_out=hidden_dims[1])
        self.Conv3 = BasicConvBlock(ch_in=hidden_dims[1], ch_out=hidden_dims[2])
        self.Conv4 = BasicConvBlock(ch_in=hidden_dims[2], ch_out=hidden_dims[3])
        self.Conv5 = BasicConvBlock(ch_in=hidden_dims[3], ch_out=hidden_dims[4])

        # decoder
        self.Up5 = UpConvBlock(ch_in=hidden_dims[4], ch_out=hidden_dims[3])
        self.Up4 = UpConvBlock(ch_in=hidden_dims[3], ch_out=hidden_dims[2])
        self.Up3 = UpConvBlock(ch_in=hidden_dims[2], ch_out=hidden_dims[1])
        self.Up2 = UpConvBlock(ch_in=hidden_dims[1], ch_out=hidden_dims[0])
        self.Up1 = nn.Conv2d(hidden_dims[0], img_channels, kernel_size=1, stride=1, padding=0)

        self.encoder = nn.Sequential(*[self.Conv1, self.MaxPool, self.Conv2, self.MaxPool,
                                       self.Conv3, self.MaxPool, self.Conv4, self.MaxPool, self.Conv5, nn.Flatten(),
                                       nn.Linear(392, coding_size)])
        self.decoder = nn.Sequential(
            *[nn.Linear(coding_size, 392), nn.Unflatten(1, (2, 14, 14)), self.Up5, self.Up4, self.Up3, self.Up2,
              self.Up1])
        init_weights(self.encoder)
        init_weights(self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvAutoEncoderTwo(nn.Module):

    def __init__(self):
        super(ConvAutoEncoderTwo, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(3, 16, 4, padding=1)),
            ('relu_1', nn.ReLU(True)),
            ('pooling_1', nn.MaxPool2d(2, 2)),
            ('conv_2', nn.Conv2d(16, 4, 3, padding=1)),
            ('relu_2', nn.ReLU(True)),
            ('pooling_2', nn.MaxPool2d(2, 2)),
            ('conv_3', nn.Conv2d(4, 4, 3, padding=1)),
            ('relu_3', nn.ReLU(True)),
            ('pooling_3', nn.MaxPool2d(2, 2)),
            ('conv_4', nn.Conv2d(4, 4, 3, padding=1)),
            ('relu_4', nn.ReLU(True)),
            ('pooling_4', nn.MaxPool2d(2, 2)),
            ('flatten', nn.Flatten()),
            ('dense', nn.Linear(4 * 13 * 13, 10))
        ]))

        self.decoder = nn.Sequential(
            nn.Linear(10, 4 * 13 * 13),
            nn.Unflatten(1, (4, 13, 13)),
            nn.MaxUnpool2d(self.encoder.pooling_4.size()),
            nn.Conv2d(4, 4, 3, stride=2),
            nn.ReLU(True),
            nn.MaxUnpool2d(self.encoder.pooling_3.size()),
            nn.Conv2d(4, 4, 3, stride=2),
            nn.ReLU(True),
            nn.MaxUnpool2d(self.encoder.pooling_2.size()),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(True),
            nn.MaxUnpool2d(self.encoder.pooling_1.size()),
            nn.Conv2d(3, 16, 4, padding=1)
        )

    def forward(self, x):
        # x = self.encoder(x)
        return self.encoder(x)


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


if __name__ == "__main__":
    # model = ConvAutoEncoderTwo()
    # summary(model.cuda(), input_size=(3, 224, 224))
    cp = Checkpoint(dirname='segnet_mse_no_sigmoid_sgd_150ep_b8_lr_0.01_30enc/checkpoints')
    train_end_cp = TrainEndCheckpoint(dirname='segnet_mse_no_sigmoid_sgd_150ep_b8_lr_0.01_30enc/checkpoints')
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
        callbacks=[cp, train_end_cp]
    )
    net.initialize()
    net.load_params(checkpoint=cp)
    mean = np.array([0.5020, 0.4690, 0.4199])
    std = np.array([0.2052, 0.2005, 0.1966])
    inverse_transform = transforms.Normalize(
        mean=(-mean) / std,
        std=1 / std
    )
    transform = transforms.Compose([transforms.Resize((224, 224),
                                                      interpolation=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                    ])
    for i in range(10):
        img = transform(Image.open(f"../cropped_data/000{i}.png").convert('RGB'))
        input_image = (img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)).numpy().transpose(1, 2,
                                                                                                                   0)
        # plt.imshow(inv_normalize(img).numpy().transpose(1, 2, 0))
        # plt.show()
        img = img.unsqueeze(0).cuda()
        decoded = net.module_(img)
        output_image = (decoded.detach().cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1,
                                                                                                           1)).numpy().squeeze(
            0).transpose(1, 2, 0)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(input_image)
        axs[1].imshow(output_image)
        plt.show()
        # plt.savefig(f"../plots/segnet/50_enc/000{i}.png")
