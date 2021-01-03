from PIL import Image
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from AutoEncoder import ConvAutoEncoder


class AutoEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, auto_encoder_model: ConvAutoEncoder, torch_transformers):
        self.torch_transformers = torch_transformers
        self.auto_encoder_model = auto_encoder_model

    def fit(self, X: pd.DataFrame, y=None):
        X.merge(X.apply(lambda row: self.__encode_images__(row.file_path), axis='columns', result_type='expand'),
                left_index=True, right_index=True)
        return self

    def transform(self, X: DataFrame, y=None):
        return self

    def __encode_images__(self, image_path, *args, **kwargs):
        img = Image.open(image_path).convert('RGB')
        tensor = self.torch_transformers(img).unsqueeze(0).cuda()
        encoder_output = self.auto_encoder_model.encoder(tensor)
        return encoder_output.detach().cpu().squeeze(0).numpy()
