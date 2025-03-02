# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import re
import cv2
from tqdm import tqdm
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

FMOW_PATH = os.getenv('FMOW_PATH')
PATCH_SIZE = os.getenv('PATCH_SIZE')

def cut_patches(img, data_dir, img_name, patch_size):
    h, w, _ = img.shape
    for i in tqdm(range(0, h, patch_size)):
        for j in tqdm(range(0, w, patch_size), leave = False):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size, 3):
                patch_name = f"{img_name}_{i}_{j}.jpg"
                patch_path = os.path.join(data_dir, patch_name)
                cv2.imwrite(patch_path, patch)

def build_fmow_dataset(fmow_path, data_dir, patch_size = PATCH_SIZE):
    """
    Cut 512x512 and copy image patches from fmow dataset to data_dir
    """
    # airport, crop_field, etc...
    categories = os.listdir(fmow_path)
    for cat in categories:
        cat_path = os.path.join(fmow_path, cat)
        cat_folders = os.listdir(cat_path)
        # airport_0, airport_1 ...
        for cat_folder in cat_folders:
            cat_folder_path = os.path.join(cat_path, cat_folder)
            cat_folder_data = os.listdir(cat_folder_path)
            # airport_0_0.jpg, airport_0_1.jpg ...
            for img_name in cat_folder_data:
                img_path = os.path.join(cat_folder_path, img_name)
                if re.search(r"\.jpg$", img_path, re.IGNORECASE):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cut_patches(img, data_dir, img_name, patch_size)    

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def main():
    fmow_path = '/Users/HP/Documents/GitHub/bldng-dmg-assess/data/fmow_test'
    data_dir = '/Users/HP/Documents/GitHub/bldng-dmg-assess/data/fmow_data'
    build_fmow_dataset(fmow_path, data_dir, 512)

if __name__ == "__main__":
    main()