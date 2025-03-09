import rasterio
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling
from rasterio.windows import Window
from skimage import exposure
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from config.constants import *
import os
import gc 
import psutil
import logging
import argparse
from tqdm import tqdm
 
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

SAR_DATA_FOLDER = os.getenv('SAR_DATA_FOLDER')
GHSL_DATA_FOLDER = os.getenv('GHSL_DATA_FOLDER')
DATA_FOLDER = os.getenv('DATA_FOLDER')
GHSL = os.getenv('GHSL')
PATCH_FOLDER = os.getenv('PATCH_DATA_FOLDER')

# Set up logging - only INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def detect_buildings(ghsl_path, sar, threshold = BUILDINGS_THRESHOLD):
    with rasterio.open(ghsl_path) as ghsl:
        # Get bounds of SAR image in GHSL CRS(WGS84)
        wgs84_bounds = transform_bounds(sar.crs, ghsl.crs, *sar.bounds)
        logging.info(f"Starting GHSL processing - Memory: {get_memory_mb():.0f}MB")

        window = from_bounds(*wgs84_bounds, ghsl.transform)

        logging.info(f"After GHSL read - Memory: {get_memory_mb():.0f}MB")
        
        window_size_gb = (window.width * window.height * 2) / (1024 * 1024 * 1024)  # assuming uint16
        logging.info(f"Estimated window size: {window_size_gb:.2f}GB")

        if window_size_gb > 1.0:  # If window would be larger than 1GB
            logging.info("Large window detected - reading in chunks")
            # Read in chunks of 10000 rows
            chunk_height = 10000
            chunks = []
            
            for row_start in range(0, window.height, chunk_height):
                chunk_window = Window(window.col_off, 
                                   window.row_off + row_start,
                                   window.width,
                                   min(chunk_height, window.height - row_start))
                                   
                logging.info(f"Reading chunk at row {row_start}/{window.height} - Memory: {get_memory_mb():.0f}MB")
                chunk = ghsl.read(1, window=chunk_window)
                chunks.append(chunk)
                
            ghsl_subset = np.vstack(chunks)
            del chunks
            gc.collect()
        else:
            ghsl_subset = ghsl.read(1, window=window)
        
        ghsl_resampled = np.zeros((sar.height, sar.width), dtype = np.uint16)

        # Reproject GHSL subset to match SAR image
        reproject(
            source=ghsl_subset,
            destination=ghsl_resampled,
            src_transform=ghsl.window_transform(window),
            src_crs=ghsl.crs,
            dst_transform=sar.transform,    
            dst_crs=sar.crs,  # Now matches SAR image CRS
            resampling=Resampling.nearest
        )
        logging.info(f"After reprojection - Memory: {get_memory_mb():.0f}MB")

        no_data = ghsl.nodata
        valid_mask = (ghsl_resampled != no_data)
        valid_pixel_count = np.count_nonzero(valid_mask)

        mask = (ghsl_resampled > threshold) & valid_mask
        building_pixel_count = np.count_nonzero(mask)
        
        # Clean up large arrays
        del ghsl_subset
        del ghsl_resampled
        del valid_mask
        del mask
        gc.collect()
        
    return (building_pixel_count / valid_pixel_count) if valid_pixel_count > 0 else 0.0

def contrast_stretch(img, lower_percent=0.1, upper_percent=99.5, gamma = 1.8):
    # Clip pixel values based on lower/upper percentiles (exclude extreme pixel values)
    lower, upper = np.percentile(img, (lower_percent, upper_percent))
    # Apply contrast stretch
    img = exposure.rescale_intensity(img, in_range=(lower, upper))
    # Apply CLAHE - tile the image and apply contrast stretching on each tile (local contrast stretching)
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    # Make image a bit darker
    img = exposure.adjust_gamma(img, gamma)
    return img

def too_much_one_col(patch, max_ratio = MAX_RATIO):
    unique, counts = np.unique(patch.flatten(), return_counts = True)
    return np.max(counts) / patch.size > max_ratio

def cut_patches(img, img_name, dir, patch_size = PATCH_SIZE, max_ratio = MAX_RATIO):
    logging.info(f"Starting patches for {img_name} - Memory: {get_memory_mb():.0f}MB")
    img_width, img_height = img.shape
    patches_processed = 0

    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            window = Window(j, i, patch_size, patch_size)
            patch = img.read(1, window = window)
            if patch.shape != (patch_size, patch_size):
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), 
                                       (0, patch_size - patch.shape[1])), mode='constant', constant_values=0)
                
            # If patch doesn't have too many of one pixel, add it to the list
            if np.sum(patch == 0) / patch.size < max_ratio:
                # Save patch to disk
                patch_path = f'{dir}/{img_name}_patch_{i}_{j}.tif'
                patch = contrast_stretch(patch)
                with rasterio.open(patch_path, 'w', driver='GTiff', width=patch_size, height=patch_size, count=1, dtype=np.float32, crs=img.crs, transform=img.window_transform(window)) as dst:
                    dst.write(patch, 1)
                
                patches_processed += 1
                if patches_processed % 1000 == 0:
                    gc.collect()
                    logging.info(f"Processed {patches_processed} patches - Memory: {get_memory_mb():.0f}MB")
                
                del patch
    gc.collect()    


def process_sar():
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    ghsl_path = f'{GHSL_DATA_FOLDER}/{GHSL}'
    
    print('Processing SAR images')
    for year in years:
        dir_names = os.listdir(f'{SAR_DATA_FOLDER}/{year}')
        for dir_name in dir_names:
            if 'geo' in dir_name.lower():
                img_path = f'{SAR_DATA_FOLDER}/{year}/{dir_name}/{dir_name}.tif'
                logging.info(f"\nProcessing {img_path}")
                logging.info(f"Initial memory: {get_memory_mb():.0f}MB")
                
                sar = rasterio.open(img_path)
                building_ratio = detect_buildings(ghsl_path, sar, threshold = BUILDINGS_THRESHOLD)
                print(f"Total building coverage for {dir_name}: {building_ratio}")
                if building_ratio >= MIN_BUILDING_COVG:
                    cut_patches(sar, dir_name, f'{DATA_FOLDER}/sar_patches')
                sar.close()
                gc.collect()
                logging.info(f"Final memory: {get_memory_mb():.0f}MB")
 
def convert_sar(sar_dir, dir_to_save):
<<<<<<< HEAD
    saved_imgs = set(os.listdir(dir_to_save))
    for img_name in tqdm(os.listdir(sar_dir)):
        img_path = f'{sar_dir}/{img_name}'
        if img_name.replace(".tif", ".png") not in saved_imgs:
            tif_image = Image.open(img_path)
            
            # Norm to RGB
            np_img = np.array(tif_image)
            np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255
            np_img = np_img.astype(np.uint8)  # Convert to 8-bit

            jpeg_image = Image.fromarray(np_img)# Convert and save as JPEG
            jpeg_image.save(f'{dir_to_save}/{img_name.replace(".tif", ".png")}', 'PNG')

def main():
    parser = argparse.ArgumentParser(description="Convert SAR images to PNG format.")
    parser.add_argument('--sar_dir', type=str, required=True, help='Directory containing SAR images.')
    parser.add_argument('--dir_to_save', type=str, required=True, help='Directory to save the converted images.')
    args = parser.parse_args()
    convert_sar(args.sar_dir, args.dir_to_save)

if __name__ == '__main__':
    main()
