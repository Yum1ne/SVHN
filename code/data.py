import os
import json
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SVHNDataset(Dataset):
    def __init__(self, path, split='train', transform=transforms.Compose(
        [transforms.Resize((64, 128)), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
        if split == 'train':
            split = 'mchar_train'
        elif split == 'val':
            split = 'mchar_val'
        else:
            raise ValueError('split must be train or val.')
        self.path = os.path.join(path, split)
        self.images = list()
        self.labels = dict()
        self.transform = transform
        with open(self.path + '.json', 'r') as f:
            train_json = json.load(f)
            for file in os.listdir(self.path):
                if file.endswith('.png'):
                    self.images.append(file)
                    self.labels[file] = train_json[file]['label']

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.images[index])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.labels[self.images[index]], dtype=np.int32)
        lbl = list(lbl) + (5 - len(lbl)) * [0]
        return torch.tensor(lbl, dtype=torch.int), img

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = SVHNDataset('tcdata', split='train')
    print(dataset[439])
