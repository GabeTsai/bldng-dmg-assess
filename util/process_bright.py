import os
from huggingface_hub import create_repo, repo_exists
from datasets import Dataset, DatasetDict, config, Features, Image, Value

import argparse

TRAIN_DIS = set(['bata-explosion', 'la_palma-volcano', 'turkey-earthquake', 'beirut-explosion', 'congo-volcano', 'haiti-earthquake', 'hawaii-wildfire'])
VAL_DIS = set(['libya-flood', 'morroco-earthquake'])

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
        os.move(pre_img_path, new_pre_img_path)

        post_img_path = os.path.join(bright_dir, post_dir, post_img)
        new_post_img_path = os.path.join(split_path, 'images', post_img)
        os.move(post_img_path, new_post_img_path)
        
        new_target_img = target_img[:target_img.find('_building_damage')] + '_target.tif'
        target_img_path = os.path.join(bright_dir, target_dir, target_img)
        new_target_img_path = os.path.join(split_path, 'targets', new_target_img)
        os.move(target_img_path, new_target_img_path)

def upload_to_hf(data_dir, repo_name = "BRIGHT-XView2Format"):
    if not repo_exists(repo_name, repo_type='dataset'):
        create_repo(repo_name, repo_type="dataset", private=False)
        print("balls")
    else:
        print(f"Repo {repo_name} already exists.")
    
    pairs = {}
    for dataset in os.listdir(data_dir):
        images_dir = os.path.join(data_dir, dataset, 'images')
        targets_dir = os.path.join(data_dir, dataset, 'targets')
        pairs[dataset] = []
        for img_name in zip(sorted(os.listdir(images_dir)), sorted(os.listdir(targets_dir))):
            img_path = os.path.join(images_dir, img_name)
            target_path = os.path.join(targets_dir, img_name)
            pairs[dataset].append({'image': img_path, 'target': target_path})
    
    features = Features({
        'image': Image(),
        'target': Image()
    })

    dataset = DatasetDict({
        'train': Dataset.from_list(pairs[dataset], features=features),
        'val': Dataset.from_list(pairs[dataset], features=features)
    }) 

    dataset.push_to_hub(repo_name, private=False)

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
