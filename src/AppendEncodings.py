import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from AePipeline import net, cp, SIZE

if __name__ == "__main__":

    features = ['width_painting_cm', 'height_painting_cm', 'width_frame_cm', 'height_frame_cm', 'condition',
                'technique', 'signed', 'framed', 'period', 'style', 'subject', 'name', 'file_path', 'price']

    data = pd.read_csv("../data preprocessing/no_nan_kunstveiling_nl_txmm.csv", header=0, usecols=features)

    net.initialize()
    net.load_params(checkpoint=cp)

    mean = np.array([0.5020, 0.4690, 0.4199])
    std = np.array([0.2052, 0.2005, 0.1966])

    torch_transformers = transforms.Compose([
        transforms.Resize(SIZE, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    for idx, row in data.iterrows():
        img = Image.open(row['file_path']).convert('RGB')
        tensor = torch_transformers(img).unsqueeze(0).cuda()
        encoder_output = net.module_.encoder(tensor)
        encoded = encoder_output.detach().cpu().squeeze(0).numpy()
        for i in range(len(encoded)):
            data[f'encoded_{i}'] = encoded[i]
    data.to_csv("../data preprocessing/final_df_with_encodings.csv")
