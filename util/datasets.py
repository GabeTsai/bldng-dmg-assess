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
from PIL import Image
import re
import cv2
from tqdm import tqdm
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import argparse
import numpy as np
import random
import tifffile as tiff

from huggingface_hub import create_repo, repo_exists
from datasets import Dataset, DatasetDict, config, Features, Image, Value

FMOW_PATH = os.getenv('FMOW_PATH')
PATCH_SIZE = os.getenv('PATCH_SIZE')

def cut_patches(img, data_dir, img_name, patch_size, save_perc):
    h, w, _ = img.shape
    count = 0
    num_saved = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape == (patch_size, patch_size, 3) and count % int(1/save_perc) == 0:
                patch_name = f"{img_name}_{i}_{j}.jpg"
                patch_path = os.path.join(data_dir, patch_name)
                cv2.imwrite(patch_path, patch)
            count += 1

def build_fmow_dataset(fmow_path, data_dir, patch_size = PATCH_SIZE, save_perc = 0.1):
    """
    Cut 512x512 and copy image patches from fmow dataset to data_dir
    """
    # airport, crop_field, etc...
    categories = os.listdir(fmow_path)
    for cat in tqdm(categories):
        cat_path = os.path.join(fmow_path, cat)
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)
        cat_folders = os.listdir(cat_path)
        # airport_0, airport_1 ...
        for cat_folder in tqdm(cat_folders):
            cat_folder_path = os.path.join(cat_path, cat_folder)
            cat_folder_data = os.listdir(cat_folder_path)
            # airport_0_0.jpg, airport_0_1.jpg ...
            for img_name in cat_folder_data:
                img_path = os.path.join(cat_folder_path, img_name)
                if re.search(r"\.jpg$", img_path, re.IGNORECASE):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cut_patches(img, cat_dir, img_name, patch_size, save_perc)    

def calculate_mean_std(data_dir, tif = False):
    """
    Get mean and std of RGB channels in dataset
    """
    imgs = random.sample(os.listdir(data_dir), 50000)
    if tif:
        sums = np.zeros(1)
        sq_sum = np.zeros(1)
    else:    
        sums = np.zeros(3)
        sq_sum = np.zeros(3)
    print(sums.shape)
    num_imgs = len(imgs)
    num_pixels = num_imgs * 512 * 512
    img_paths = []
    for img_name in tqdm(imgs):
        path = os.path.join(data_dir, img_name)
        if 'tif' in img_name:
            img = tiff.imread(path) 
        else:
            img = cv2.imread(path) # H, W, C
            img = img/ 255.0
        sums += np.sum(img, axis=(0, 1))
        img_paths.append(path)

    mean = sums / num_pixels
    print(f"Mean: {mean}")
    for img_path in tqdm(img_paths, desc="Computing std"):
        if 'tif' in img_name:
            img = tiff.imread(img_path)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

        sq_sum += ((img - mean) ** 2).sum(axis=(0, 1))

    return mean, np.sqrt(sq_sum / (num_pixels - 1))

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
    parser = argparse.ArgumentParser(description="Dataset processing script.")
    subparsers = parser.add_subparsers(dest='command')
    
    parser_cut = subparsers.add_parser('cut_patches', help='Cut patches from FMoW dataset images.')
    parser_cut.add_argument('--fmow_path', type=str, required=True, help='Path to the FMoW dataset.')
    parser_cut.add_argument('--data_dir', type=str, required=True, help='Directory to save the patches.')
    parser_cut.add_argument('--patch_size', type=int, default=512, help='Size of the patches.')
    parser_cut.add_argument('--save_percentage', type=float, default=0.1, help='Percentage of patches to save.')

    parser_mean_std = subparsers.add_parser('mean_std', help='Calculate mean and std of dataset.')
    parser_mean_std.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser_mean_std.add_argument('--tif', action = 'store_true', help = 'Processing tif images')

    args = parser.parse_args() 
    if args.command == 'cut_patches':
        build_fmow_dataset(args.fmow_path, args.data_dir, args.patch_size, args.save_percentage)
    elif args.command == 'mean_std':
        mean, std = calculate_mean_std(args.data_dir, args.tif)
        print(f"Mean: {mean}, Std: {std}")
    
if __name__ == "__main__":
    main()
