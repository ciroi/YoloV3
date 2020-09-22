import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from utils.augmentations import mosaic
from utils.augmentations import gridmask
from torch.utils.data import Dataset
import torchvision.transforms as transforms

aug_dic = {
    "NONE": 0,
    "GRIDMASK": 1,
    "MOSAIC": 2,
    "BOTH": 3,
}

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True, mode=aug_dic["BOTH"], maxepoch=1):
        with open(list_path, "r") as file: 
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.mode = mode
        self.maxepoch = maxepoch

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        if self.augment and np.random.random() < 0.5 and (aug_dic["MOSAIC"] == self.mode or aug_dic["BOTH"] == self.mode):
            # Load mosaic
            len_ = len(self.img_files)
            indices = [index] + [random.randint(0, len_ - 1) for _ in range(3)]  # 3 additional image indices
            
            images = []
            targets = []
            img_path = ""
            is_first = True
            for indice in indices:
                img_path_ = self.img_files[indice % len_].rstrip()
                if is_first:
                    img_path = img_path_
                    is_first = False
                img = transforms.ToTensor()(Image.open(img_path_).convert('RGB'))
                # Handle images with less than three channels
                if len(img.shape) != 3:
                    img = img.unsqueeze(0)
                    img = img.expand((3, img.shape[1:]))
                
                label_path = self.label_files[indice % len_].rstrip()
                if os.path.exists(label_path):
                    boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

                images.append(img)
                targets.append(boxes)

            img, targets_ = mosaic(images, targets, self.img_size)
            targets = torch.zeros((targets_.size()[0], 6))
            targets[:, 1:] = targets_
        else:
            img_path = self.img_files[index % len(self.img_files)].rstrip()

            # Extract image as PyTorch tensor
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

            # Handle images with less than three channels
            if len(img.shape) != 3:
                img = img.unsqueeze(0)
                img = img.expand((3, img.shape[1:]))
            
            _, h, w = img.shape

            if self.augment and aug_dic["GRIDMASK"] == self.mode or aug_dic["BOTH"] == self.mode:
                epoch = np.random.randint(self.maxepoch)
                prob = min(1, epoch / self.maxepoch)
                if np.random.random() < prob:
                    img = gridmask(img, mode=1, rotate=1, r_ratio=0.4, d_ratio=0.25)

            h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
            # Pad to square resolution
            img, pad = pad_to_square(img, 0)
            _, padded_h, padded_w = img.shape

            # ---------
            #  Label
            # ---------

            label_path = self.label_files[index % len(self.img_files)].rstrip()

            targets = None
            if os.path.exists(label_path):
                boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
                # Extract coordinates for unpadded + unscaled image
                x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
                y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
                x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
                y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
                # Adjust for added padding
                x1 += pad[0]
                y1 += pad[2]
                x2 += pad[1]
                y2 += pad[3]
                # boxes[:, 0] => class id
                # Returns (xc, yc, w, h)
                boxes[:, 1] = ((x1 + x2) / 2) / padded_w        # newer center x
                boxes[:, 2] = ((y1 + y2) / 2) / padded_h        # newer center y
                boxes[:, 3] *= w_factor / padded_w              # newer width
                boxes[:, 4] *= h_factor / padded_h              # newer height

                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
