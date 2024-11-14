import torch
from torch.utils.data import Dataset

from sprites_datagen.moving_sprites import MovingSpriteDataset

class SpritesDataset(Dataset):

    def __init__(self, spec, dataset_size):
        self.spec = spec
        self.dataset_size = dataset_size

        self.data = []
        for i in range(self.dataset_size):
            datapoint = MovingSpriteDataset(spec).__getitem__(i)
            self.data.append(datapoint)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


