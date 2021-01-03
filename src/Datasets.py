import glob

from PIL import Image
from torch.utils.data import Dataset


class AutoEncoderImageDataset(Dataset):

    def __init__(self, root_dir, transform, file_ext="png"):
        self.file_ext = file_ext
        self.transform = transform
        self.root_dir = root_dir
        self.img_paths = glob.glob(f"{self.root_dir}/*.{self.file_ext}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = Image.open(img_path).convert('RGB')
        if self.transform:
            sample = self.transform(sample)

        return sample, sample
