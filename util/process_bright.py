import os
import shutil
from huggingface_hub import create_repo, repo_exists
from datasets import Dataset, DatasetDict, config, Features, Image, Value
from huggingface_hub.utils import logging as hfhub_logging
import tifffile as tiff

import argparse

from tqdm import tqdm

import psutil

hfhub_logging.set_verbosity_debug()

TRAIN_DIS = set(['bata-explosion', 'la_palma-volcano', 'turkey-earthquake', 'beirut-explosion', 'congo-volcano', 'haiti-earthquake', 'hawaii-wildfire'])
VAL_DIS = set(['libya-flood', 'morocco-earthquake'])

CLASSES = ('explosion', 'volcano', 'earthquake', 'wildfire', 'flood')

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

def create_class_finetune(bright_dir, data_dir):
    """
    Create classes from bright dataset to evaluate model via finetuning.
    Args:
        bright_dir (str): Directory containing the bright dataset.
        data_dir (str): Directory to save the rearranged dataset.
    """
   
    # Create the necessary directories in data_dir for the train and val splits
    for split in ('train', 'val'):
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for event in CLASSES:
            if not os.path.exists(os.path.join(split_dir, event)):
                os.makedirs(os.path.join(split_dir, event))
    
    for split in ('train', 'val'):
        if not os.path.exists(os.path.join(data_dir, split)):
            os.makedirs(os.path.join(data_dir, split)) 
        bright_split_dir = os.path.join(bright_dir, split)
        for img_name in os.listdir(os.path.join(bright_split_dir, 'post-event')):
            
            event_name = ''
            for disaster in CLASSES:
                if disaster in img_name:
                    event_name = disaster
                    break
            
            img_path = os.path.join(data_dir, split, event_name, img_name)
            shutil.move(os.path.join(bright_split_dir, 'post-event', img_name), img_path)

def rearrange_bright(bright_dir, data_dir, split):
    """
    Reorganize bright dataset split folder following torchange xview2 structure.

    Args:
        bright_dir (str): Directory containing the split for the bright dataset.
        data_dir (str): Directory to save the rearranged dataset.
    """
    if split not in ['train', 'val', 'test']: raise ValueError("Invalid split")
 
    if not os.path.exists(os.path.join(data_dir, split)):
        create_split_dirs(data_dir, split)
    print(os.listdir(bright_dir))
    post_dir, pre_dir, target_dir = sorted(os.listdir(bright_dir))
    
    for pre_img, post_img, target_img in zip(
        sorted(os.listdir(os.path.join(bright_dir, pre_dir))),
        sorted(os.listdir(os.path.join(bright_dir, post_dir))),
        sorted(os.listdir(os.path.join(bright_dir, target_dir)))
        ):
        disaster = target_img[:target_img.find('_0')]
        
        split_path = os.path.join(data_dir, split)
        
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
    os.chdir(data_dir)
    assert os.path.exists("train/images/bata-explosion_00000000_pre_disaster.tif"), "Image path not valid" 
    if not repo_exists(repo_name, repo_type='dataset'):
        create_repo(repo_name, repo_type="dataset", private=False)
    else:
        print(f"Repo {repo_name} already exists.")
    
    splits = {}
    for dataset in tqdm(('train', 'val', 'test')):
        images_dir = os.path.join(data_dir, dataset, 'images')
        targets_dir = os.path.join(data_dir, dataset, 'targets')
        print(images_dir, targets_dir)
        splits[dataset] = []
        for target_name in tqdm(sorted(os.listdir(targets_dir))):
            event_full_name = target_name[:target_name.find('_target')]
            event_name = target_name[:target_name.find('_0')]
            pre_img_path = os.path.join(images_dir, event_full_name + '_pre_disaster.tif') 
            post_img_path = os.path.join(images_dir, event_full_name + '_post_disaster.tif')
            target_path = os.path.join(targets_dir, target_name)
            
            pre_img_rel_path = os.path.relpath(pre_img_path, start = data_dir) 
            post_img_rel_path = os.path.relpath(post_img_path, start = data_dir) 
            target_img_rel_path = os.path.relpath(target_path, start = data_dir)
            if not (os.path.exists(os.path.join(data_dir, pre_img_rel_path)) and
                os.path.exists(os.path.join(data_dir, post_img_rel_path)) and
                os.path.exists(os.path.join(data_dir, target_img_rel_path))):
                raiseValueError("Path(s) do not exist")
            #pre_img = tiff.imread(pre_img_path)
            #post_img = tiff.imread(post_img_path)
            #target = tiff.imread(target_path)
            #
            #splits[dataset].append({'t1_image': pre_img, 't2_image': post_img, 
            #                        'change_mask': target, 'image_name': f"{dataset}/images/{event_full_name}", 
            #                        'event_name': event_name})
            #process = psutil.Process(os.getpid())
            splits[dataset].append({
                't1_image': pre_img_rel_path,
                't2_image': post_img_rel_path,
                'change_mask': target_img_rel_path,
                'image_name': f"{dataset}/images/{event_full_name}",
                'event_name': event_name
            })

            #mem_gb = process.memory_info().rss / 1024**3  # Convert bytes to GB
            #
            #tqdm.write(f" Memory usage: {mem_gb:.2f} GB")    
    features = Features({
        't1_image': Image(),
        't2_image': Image(), 
        'change_mask': Image(), 
        'image_name' : Value(dtype='string'),
        'event_name': Value(dtype='string')
    })

    dataset_hf = DatasetDict({
        'train': Dataset.from_list(splits['train'], features=features),
        'val': Dataset.from_list(splits['val'], features=features),
        'test': Dataset.from_list(splits['test'], features=features)        
    }) 

    dataset_hf.push_to_hub(repo_name, private=False)

def main():
    parser = argparse.ArgumentParser(description="Uploading BRIGHT w/ XView2 Format to HF.")
    subparsers = parser.add_subparsers(dest='command')

    parser_rearrange_bright = subparsers.add_parser('rearrange_bright', help='Rearrange bright dataset to follow torchange xview2 structure')
    parser_rearrange_bright.add_argument('--bright_dir', type=str, required=True, help='Directory containing the bright dataset.')
    parser_rearrange_bright.add_argument('--data_dir', type=str, required=True, help='Directory to save the rearranged dataset.')    
    parser_rearrange_bright.add_argument('--split', type=str, required=True, help='Data split')

    parser_upload = subparsers.add_parser('upload_to_hf', help='Upload dataset to Hugging Face Hub.')
    parser_upload.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser_upload.add_argument('--repo_name', type=str, default="BRIGHT-XView2Format", help='Name of the Hugging Face repository.')

    parser_create_finetune = subparsers.add_parser('create_class_finetune', help='Create classes from bright dataset to evaluate model via finetuning.')
    parser_create_finetune.add_argument('--bright_dir', type=str, required=True, help='Directory containing the bright dataset.')
    parser_create_finetune.add_argument('--data_dir', type=str, required=True, help='Directory to save the finetune dataset.')

    args = parser.parse_args()
    
    if args.command == 'rearrange_bright':
        rearrange_bright(args.bright_dir, args.data_dir, args.split)
    elif args.command == 'upload_to_hf':
        upload_to_hf(args.data_dir, args.repo_name)
    elif args.command == 'create_class_finetune':
        create_class_finetune(args.bright_dir, args.data_dir)

if __name__ == "__main__":
    main()
