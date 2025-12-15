import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ResizeAndPad:
    def __init__(self, target_size=(64, 512), fill=255):
        """
        等比例缩放图片后填充到目标尺寸
        :param target_size: 目标尺寸 (height, width)，此处为(64, 512)
        :param fill: 填充的像素值，默认白色（255）
        """
        self.target_h, self.target_w = target_size
        self.fill = fill

    def __call__(self, img):
        # 获取图片原始尺寸
        orig_h, orig_w = img.size[1], img.size[0]  # PIL.Image的size是(w, h)

        # 计算缩放比例：保持宽高比，缩放到目标尺寸的最大适配比例
        scale_h = self.target_h / orig_h
        scale_w = self.target_w / orig_w
        scale = min(scale_h, scale_w)  # 取最小比例，避免超出目标尺寸

        # 计算缩放后的尺寸
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        # 等比例缩放图片
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)  # 高质量缩放

        # 计算填充的边距（居中填充）
        pad_top = (self.target_h - new_h) // 2
        pad_bottom = self.target_h - new_h - pad_top
        pad_left = (self.target_w - new_w) // 2
        pad_right = self.target_w - new_w - pad_left

        # 填充图片到目标尺寸（PIL的expand方法）
        img = Image.new(img.mode, (self.target_w, self.target_h), self.fill)
        img.paste(img.resize((new_w, new_h)), (pad_left, pad_top))

        return img


class SVHNDataset(Dataset):
    def __init__(self, path, split='train', transform=transforms.Compose(
        [transforms.Grayscale(num_output_channels=1), ResizeAndPad(target_size=(64, 512), fill=255),
         transforms.ToTensor(), transforms.Normalize([0.485], [0.229])])):
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

        lbl = self.labels[self.images[index]]
        lbl = lbl + (6 - len(lbl)) * [10]
        return torch.tensor(lbl, dtype=torch.int), img

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = SVHNDataset('../tcdata', split='train')
    print(dataset[44])
