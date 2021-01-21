import numpy as np
import pandas as pd
import torch
from PIL import Image
from skorch import NeuralNetRegressor
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, LoadInitState
from torch.nn import MSELoss

from torchvision import transforms
from AePipeline import net, cp, SIZE
from ConvAutoEncoder import SegNet

if __name__ == "__main__":

    features = ['width_painting_cm', 'height_painting_cm', 'width_frame_cm', 'height_frame_cm', 'condition',
                'technique', 'signed', 'framed', 'period', 'kunstenaar', 'style', 'subject', 'name', 'file_path', 'price',
                'price_binned']

    data = pd.read_csv("../data preprocessing/no_nan_binned_prices_kunstveiling_nl_txmm.csv", header=0,
                       usecols=features)

    SIZE = (224, 224)  # Resize the image to this shape
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

    net.initialize()
    net.load_params(checkpoint=cp)

    mean = np.array([0.5020, 0.4690, 0.4199])
    std = np.array([0.2052, 0.2005, 0.1966])
    torch_transformers = transforms.Compose([
        transforms.Resize(SIZE, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    encodings = []
    for idx, row in data.iterrows():
        img = Image.open(row['file_path']).convert('RGB')
        tensor = torch_transformers(img).unsqueeze(0).cuda()
        decoded = net.module_(tensor)
        encoded = net.module_.encoded.detach().cpu().squeeze(0).numpy()
        for i in range(len(encoded)):
            data.at[idx, f'encoded_{i}'] = encoded[i]
    data.to_csv("../data preprocessing/final_df_with_encodings_with_price_binned.csv", index=False)
