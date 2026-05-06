import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers

from PIL import Image
import cv2
import os
import pandas as pd

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths





class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(8, 8), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # [-1,1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name


#@Registers.datasets.register_with_name('custom_single')
@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    '''custom dataset for aligned image pairs, e.g. for super-resolution'''

    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        csv_path = dataset_config.csv_path  
        self.quality_scores = pd.read_csv(csv_path, header=None, names=["name", "score"])
        self.quality_dict = {str(row["name"]): row["score"] for _, row in self.quality_scores.iterrows()}

        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/HiRes'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/LowRes'))

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

        self.weights = []
        for path in image_paths_ori:
            filename = os.path.basename(path) #after last '/'
            base_name = os.path.splitext(filename)[0] # Extract the first 7 characters
            weight = self.quality_dict.get(base_name, 1.0)  # Default weight to 1.0 if not found
            self.weights.append(weight)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        #print(self.imgs_ori[i][0])
        #print(type(self.imgs_ori[i][0]))
        return self.imgs_ori[i][0], self.imgs_ori[i][1], self.imgs_cond[i][0], self.imgs_cond[i][1], torch.tensor(self.weights[i], dtype=torch.float32)



