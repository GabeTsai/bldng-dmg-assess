import os
import shutil
from huggingface_hub import create_repo, repo_exists
from datasets import Dataset, DatasetDict, config, Features, Image, Value
import tifffile as tiff

import argparse

TRAIN_DIS = set(['bata-explosion', 'la_palma-volcano', 'turkey-earthquake', 'beirut-explosion', 'congo-volcano', 'haiti-earthquake', 'hawaii-wildfire'])
VAL_DIS = set(['libya-flood', 'morocco-earthquake'])

def create_split_dirs(base_dir, split_name):
    """
    Create the necessary directories for a given split (e.g., train or val).
    
    Args:
        base_dir (str): The base directory where the split directories will be created.
        split_name (str): The name of the split (e.g., 'train' or 'val').
    """
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'targets'), exist_ok=True)

def rearrange_bright(bright_dir, data_dir):
    """
    Reorganize bright dataset train folder into train/val split following torchange xview2 structure.
    Split is done based on disaster, with most recent disasters in validation dataset to prevent
    temporal leakage.

    Args:
        bright_dir (str): Directory containing the bright dataset.
        data_dir (str): Directory to save the rearranged dataset.
    """
    
    create_split_dirs(data_dir, 'train')
    create_split_dirs(data_dir, 'val')
    print(os.listdir(bright_dir))
    post_dir, pre_dir, target_dir = sorted(os.listdir(bright_dir))
    
    for pre_img, post_img, target_img in zip(
        sorted(os.listdir(os.path.join(bright_dir, pre_dir))),
        sorted(os.listdir(os.path.join(bright_dir, post_dir))),
        sorted(os.listdir(os.path.join(bright_dir, target_dir)))
        ):
        disaster = target_img[:target_img.find('_0')]
        
        if disaster in TRAIN_DIS:
            split_path = os.path.join(data_dir, 'train')
        elif disaster in VAL_DIS:
            split_path = os.path.join(data_dir, 'val')
        else:
            raise ValueError(f"Unknown disaster {disaster} in {target_img}.")
        
        pre_img_path = os.path.join(bright_dir, pre_dir, pre_img)
        new_pre_img_path = os.path.join(split_path, 'images', pre_img)        
        shutil.move(pre_img_path, new_pre_img_path)

        post_img_path = os.path.join(bright_dir, post_dir, post_img)
        new_post_img_path = os.path.join(split_path, 'images', post_img)
        shutil.move(post_img_path, new_post_img_path)
        
        new_target_img = target_img[:target_img.find('_building_damage')] + '_target.tif'
        target_img_path = os.path.join(bright_dir, target_dir, target_img)
        new_target_img_path = os.path.join(split_path, 'targets', new_target_img)
        shutil.move(target_img_path, new_target_img_path)

def upload_to_hf(data_dir, repo_name = "BRIGHT-XView2Format"):
    if not repo_exists(repo_name, repo_type='dataset'):
        create_repo(repo_name, repo_type="dataset", private=False)
    else:
        print(f"Repo {repo_name} already exists.")
    
    splits = {}
    for dataset in ('train', 'val'):
        images_dir = os.path.join(data_dir, dataset, 'images')
        targets_dir = os.path.join(data_dir, dataset, 'targets')
        splits[dataset] = []
        for img_name, target_name in zip(sorted(os.listdir(images_dir)), sorted(os.listdir(targets_dir))):
            dis_name_code = ''
            if 'pre' in img_name:      
                dis_name_code = img_name[:img_name.find('_pre')]
            else:    
                dis_name_code = img_name[:img_name.find('_post')]
            pre_img_path = os.path.join(images_dir, dis_name_code + '_pre_disaster.tif') 
            post_img_path = os.path.join(images_dir, dis_name_code + '_post_disaster.tif')
            target_path = os.path.join(targets_dir, target_name)

            pre_img = tiff.imread(pre_img_path)
            post_img = tiff.imread(post_img_path)
            target = tiff.imread(target_path)

            splits[dataset].append({'t1_image': pre_img, 't2_image': post_img, 
                                    'change_mask': target, 'image_name': f"{dataset}/images/{dis_name_code}"})
    
    features = Features({
        't1_image': Image(),
        't2_image': Image(), 
        'change_mask': Image(), 
        'image_name' : Value(dtype='string')
    })

    dataset_hf = DatasetDict({
        'train': Dataset.from_list(splits['train'], features=features),
        'val': Dataset.from_list(splits['val'], features=features)
    }) 

    dataset_hf.push_to_hub(repo_name, private=False)

def main():
    parser = argparse.ArgumentParser(description="Uploading BRIGHT w/ XView2 Format to HF.")
    subparsers = parser.add_subparsers(dest='command')
    

    parser_rearrange_bright = subparsers.add_parser('rearrange_bright', help='Rearrange bright dataset to follow torchange xview2 structure')
    parser_rearrange_bright.add_argument('--bright_dir', type=str, required=True, help='Directory containing the bright dataset.')
    parser_rearrange_bright.add_argument('--data_dir', type=str, required=True, help='Directory to save the rearranged dataset.')

    parser_upload = subparsers.add_parser('upload_to_hf', help='Upload dataset to Hugging Face Hub.')
    parser_upload.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser_upload.add_argument('--repo_name', type=str, default="BRIGHT-XView2Format", help='Name of the Hugging Face repository.')

    args = parser.parse_args()
    
    if args.command == 'rearrange_bright':
        rearrange_bright(args.bright_dir, args.data_dir)
    elif args.command == 'upload_to_hf':
        upload_to_hf(args.data_dir, args.repo_name)

if __name__ == "__main__":
    main()
