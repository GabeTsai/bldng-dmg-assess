import os
from huggingface_hub import create_repo, repo_exists
from datasets import Dataset, DatasetDict, config, Features, Image, Value

import argparse


def rearrange_bright(bright_dir, data_dir):
    """
    Reorganize bright dataset to follow torchange xview2 structure

    Args:
        bright_dir (str): Directory containing the bright dataset.
        data_dir (str): Directory to save the rearranged dataset.
    """
    if not os.path.exists(data_dir + '/images'):
        os.makedirs(data_dir + '/images')
    if not os.path.exists(data_dir + '/targets'):
        os.makedirs(data_dir + '/targets')
    pre_dir, post_dir, target_dir = sorted(os.listdir(data_dir))
    for pre_img, post_img, target_img in zip(
        sorted(os.listdir(os.path.join(bright_dir, pre_dir))),
        sorted(os.listdir(os.path.join(bright_dir, post_dir))),
        sorted(os.listdir(os.path.join(bright_dir, target_dir)))
        ):
        pre_img_path = os.path.join(bright_dir, pre_dir, pre_img)
        new_pre_img_path = os.path.join(data_dir, 'images', pre_img)
        os.move(pre_img_path, new_pre_img_path)
        post_img_path = os.path.join(bright_dir, post_dir, post_img)
        new_post_img_path = os.path.join(data_dir, 'images', post_img)
        os.move(post_img_path, new_post_img_path)\
        
        new_target_img = target_img[:target_img.find('.')] + '.tif'
        target_img_path = os.path.join(bright_dir, target_dir, target_img)
        new_target_img_path = os.path.join(data_dir, 'targets', new_target_img)
        os.move(target_img_path, new_target_img_path)

def upload_to_hf(data_dir, repo_name = "BRIGHT-XView2Format"):
    if not repo_exists(repo_name, repo_type='dataset'):
        create_repo(repo_name, repo_type="dataset", private=False)
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
