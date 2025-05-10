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
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import pickle
 
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
                
            # If patch doesn't have too many black pixels, and the image has enough contrast (isn't straight up noise), add patch 
            if np.sum(patch == 0) / patch.size < max_ratio:
                log_patch = np.log10(patch, out=np.zeros_like(patch, dtype=np.float32), where=(patch!=0))
                if np.max(log_patch) - np.min(log_patch[log_patch != 0]) > 0.5:
                    patch_path = f'{dir}/{img_name}_patch_{i}_{j}.tif'
                    with rasterio.open(patch_path, 'w', driver='GTiff', width=patch_size, height=patch_size, count=1, dtype=np.float32, crs=img.crs, transform=img.window_transform(window)) as dst:
                        dst.write(log_patch, 1)
                    
                    patches_processed += 1
                    if patches_processed % 1000 == 0:
                        gc.collect()
                        logging.info(f"Processed {patches_processed} patches - Memory: {get_memory_mb():.0f}MB")
                    
                    del patch
    gc.collect()    

def plot_coverage(boxes):
    """
    Plot the coverage of the SAR images on a map.
    Arguments:
    - boxes: List of bounding boxes for the SAR images.
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi = 500)
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
    world.boundary.plot(ax=ax, linewidth=1)
    
    for box in boxes:
        left, bottom, right, top = box
        center_x = (left + right) / 2
        center_y = (bottom + top) / 2
        ax.scatter(center_x, center_y, color = 'red', s = 10, alpha = 0.5)     
    
    plt.title('SAR Image Coverage')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('sar_coverage.png')

def process_sar(sar_dir, target_dir, get_coverage):
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    ghsl_path = f'{GHSL_DATA_FOLDER}/{GHSL}'
    
    boxes = []
    used_images = set()
    if os.path.exists('used_sar_images.pkl'):
        with open('used_sar_images.pkl', 'rb') as fp:
            used_images = pickle.load(fp)

    print('Processing SAR images')
    for year in years:
        dir_names = os.listdir(f'{sar_dir}/{year}')
        for dir_name in dir_names:
            if 'geo' in dir_name.lower():
                img_path = f'{sar_dir}/{year}/{dir_name}/{dir_name}.tif'
                logging.info(f"\nProcessing {img_path}")
                logging.info(f"Initial memory: {get_memory_mb():.0f}MB")
                
                sar = rasterio.open(img_path)
                if dir_name in used_images: 
                    bounds = sar.bounds     # Get bounds of SAR image in WGS84
                    wgs84_bounds = transform_bounds(sar.crs, 'EPSG:4326',
                                bounds.left, bounds.bottom,
                                bounds.right, bounds.top)
                    print(wgs84_bounds)
                    boxes.append(wgs84_bounds)
                    continue

                building_ratio = detect_buildings(ghsl_path, sar, threshold = BUILDINGS_THRESHOLD)
                print(f"Total building coverage for {dir_name}: {building_ratio}")
                if building_ratio >= MIN_BUILDING_COVG:
                    if get_coverage:
                        bounds = sar.bounds     # Get bounds of SAR image in WGS84
                        wgs84_bounds = transform_bounds(sar.crs, 'EPSG:4326',
                                    bounds.left, bounds.bottom,
                                    bounds.right, bounds.top)
                        print(wgs84_bounds)
                        boxes.append(wgs84_bounds)
                    else:
                        cut_patches(sar, dir_name, target_dir)
                sar.close()
                gc.collect()
                logging.info(f"Final memory: {get_memory_mb():.0f}MB")
    if boxes:
        plot_coverage(boxes)
    if not os.path.exists('used_sar_images.pkl'):
        with open('used_sar_images.pkl', 'wb') as fp:
            pickle.dump(used_images, fp)

def convert_sar(sar_dir, dir_to_save):
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
    parser = argparse.ArgumentParser(description="SAR image processing utility.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser for convert_sar
    convert_parser = subparsers.add_parser('convert', help='Convert SAR TIFF images to PNG format.')
    convert_parser.add_argument('--sar_dir', type=str, required=True, help='Directory containing SAR images.')
    convert_parser.add_argument('--dir_to_save', type=str, required=True, help='Directory to save the converted images.')

    # Subparser for process_sar
    process_parser = subparsers.add_parser('process', help='Process SAR images for building detection.')
    process_parser.add_argument('--sar_dir', type=str, required=True, help='Directory containing SAR image folders.')
    process_parser.add_argument('--target_dir', type=str, required=True, help='Directory to save processed patches.')
    process_parser.add_argument('--get_coverage', action='store_true', help='Flag to get global coverage of chosen SAR images.')
    args = parser.parse_args()

    if args.command == 'convert':
        convert_sar(args.sar_dir, args.dir_to_save)
    elif args.command == 'process':
        process_sar(args.sar_dir, args.target_dir, args.get_coverage)

if __name__ == '__main__':
    main()
