import rasterio
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling
from rasterio.windows import Window
from skimage import exposure
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import config.constants as constants
import os
import gc 
import psutil
import logging
import argparse
import geopandas as gpd
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import math
import pickle
 
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

SAR_DATA_FOLDER = os.getenv('SAR_DATA_FOLDER')
GHSL_DATA_FOLDER = os.getenv('GHSL_DATA_FOLDER')
DATA_FOLDER = os.getenv('DATA_FOLDER')
GHSL_MASK = os.getenv('GHSL')
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

def plot_coverage(boxes):
    """
    Plot the coverage of the SAR images on a map.
    Arguments:
    - boxes: List of bounding boxes for the SAR images.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.boundary.plot(ax=ax, linewidth=1)
    
    for box in boxes:
        bottom, left, right, top = box
        rect = plt.Rectangle((left, bottom), right - left, top - bottom, linewidth=1, edgecolor = 'red', alpha = 0.25, facecolor = 'red')
        ax.add_patch(rect)
    
    plt.title('SAR Image Coverage')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('sar_coverage.png')

def process_sar_single_image(capella_dir, year, dir_name, ghsl, 
                             get_coverage, target_dir, boxes, patch_metadata_path, 
                             image_coverage_path):
    img_path = f'{capella_dir}/{year}/{dir_name}/{dir_name}.tif'
    logging.info(f"\nProcessing {img_path}")
    logging.info(f"Initial memory: {get_memory_mb():.0f}MB")
    
    sar = rasterio.open(img_path)
    building_ratio = detect_buildings(ghsl, sar, threshold = constants.BUILDINGS_THRESHOLD)
    print(f"Total building coverage for {dir_name}: {building_ratio}")
    if building_ratio >= constants.MIN_BUILDING_COVG:
        if get_coverage:
            bounds = sar.bounds     # Get bounds of SAR image in WGS84
            wgs84_bounds = transform_bounds(sar.crs, 'EPSG:4326',
                        bounds.left, bounds.bottom,
                        bounds.right, bounds.top)
            boxes.append(wgs84_bounds)
            with open(image_coverage_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dir_name, building_ratio])
        else:
            cut_patches(sar, dir_name, target_dir, patch_metadata_path)
        
    sar.close()
    gc.collect()
    logging.info(f"Final memory: {get_memory_mb():.0f}MB")

def process_sar(capella_dir, target_dir, get_coverage):
    ghsl_path = f'{GHSL_DATA_FOLDER}/{GHSL_MASK}'
    patch_metadata_path = f'{SAR_DATA_FOLDER}/{constants.PATCH_METADATA_FILENAME}'
    image_coverage_path = f'{SAR_DATA_FOLDER}/{constants.PATCH_BLDNG_COVERAGE_FILENAME}'
    
    with open(patch_metadata_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=constants.SAR_IMAGE_METADATA_HEADERS)
        writer.writeheader()

    boxes = []
    print('Processing SAR images')
    with rasterio.open(ghsl_path) as ghsl:
        for year in constants.SAR_YEARS:
            dir_names = os.listdir(f'{capella_dir}/{year}')
            for dir_name in dir_names:
                if 'geo' in dir_name.lower():
                    process_sar_single_image(capella_dir, year, dir_name, ghsl, get_coverage, 
                                             target_dir, boxes, patch_metadata_path, image_coverage_path)
    if get_coverage:
        plot_coverage(boxes)

def detect_buildings(ghsl, sar, threshold = constants.BUILDINGS_THRESHOLD):
    # Get bounds of SAR image in GHSL CRS(WGS84)
    wgs84_bounds = transform_bounds(sar.crs, ghsl.crs, *sar.bounds)
    logging.info(f"Starting GHSL processing - Memory: {get_memory_mb():.0f}MB")

    window = from_bounds(*wgs84_bounds, ghsl.transform)

    logging.info(f"After GHSL read - Memory: {get_memory_mb():.0f}MB")
    
    window_size_gb = (window.width * window.height * 2) / (1024 * 1024 * 1024)  # assuming uint16
    logging.info(f"Estimated window size: {window_size_gb:.2f}GB")

    if window_size_gb > 1.0: # If window would be larger than 1GB
        logging.info("Large window detected - reading in chunks")
        chunk_height = 10000 # Read in chunks of 10000 rows
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

    valid_mask = (ghsl_resampled != ghsl.nodata)
    valid_pixel_count = np.count_nonzero(valid_mask)

    mask = (ghsl_resampled > threshold) & valid_mask
    building_pixel_count = np.count_nonzero(mask)
    
    del ghsl_subset
    del ghsl_resampled
    del valid_mask
    del mask
    gc.collect()
        
    return (building_pixel_count / valid_pixel_count) if valid_pixel_count > 0 else 0.0   

def up_contrast_convert_to_uint8(img_uint16):
    img = img_uint16.astype(np.float32)

    mask = img > 0  # ignore zero values
    if not np.any(mask):
        return np.zeros_like(img, dtype=np.uint8)

    img = np.maximum(img, 1.0)
    img_db = 10 * np.log10(img)

    vmin, vmax = np.percentile(img_db[mask], (1, 99))
    img_db = np.clip(img_db, vmin, vmax)

    img_norm = np.zeros_like(img_db)
    img_norm[mask] = (img_db[mask] - vmin) / (vmax - vmin) * 255

    return img_norm.astype(np.uint8)

def cut_patches(img, img_name, sar_dir, patch_metadata_path, 
                patch_size = constants.PATCH_SIZE, max_ratio = constants.MAX_RATIO):
    logging.info(f"Starting patches for {img_name} - Memory: {get_memory_mb():.0f}MB")
    img_width, img_height = img.shape
    patches_processed = 0
    crs_is_geo = img.crs and img.crs.is_geographic

    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            window = Window(j, i, patch_size, patch_size)
            patch = img.read(1, window = window)

            center_row = i + patch_size // 2
            center_col = j + patch_size // 2

            x_center, y_center = rasterio.transform.xy(img.transform, center_row, center_col)

            if crs_is_geo:
                lon_center, lat_center = rasterio.warp.transform(
                    img.crs, "EPSG:4326", [x_center], [y_center]
                )
                lon_center, lat_center = lon_center[0], lat_center[0]
            else:
                lon_center, lat_center = x_center, y_center

            if patch.shape != (patch_size, patch_size):
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), 
                                       (0, patch_size - patch.shape[1])), mode='constant', constant_values=0)
                
            # If patch doesn't have too many black pixels, and the image has enough contrast (isn't straight up noise), add patch 
            if np.sum(patch == 0) / patch.size < max_ratio:
                log_patch = up_contrast_convert_to_uint8(patch)
                if np.max(log_patch) - np.min(log_patch[log_patch != 0]) > 0.5:
                    patch_name = f'{img_name}_patch_{i}_{j}.tif'
                    patch_path = f'{sar_dir}/{patch_name}'
                    with rasterio.open(patch_path, 'w', driver='GTiff', width=patch_size, height=patch_size, count=1, dtype=np.float32, crs=img.crs, transform=img.window_transform(window)) as dst:
                        dst.write(log_patch, 1)
                    
                    patches_processed += 1
                    if patches_processed % 1000 == 0:
                        gc.collect()
                        logging.info(f"Processed {patches_processed} patches - Memory: {get_memory_mb():.0f}MB")
                    
                    with open(patch_metadata_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([patch_name, lat_center, lon_center])
                    del patch
    gc.collect()   
 
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
    process_parser.add_argument('--get_coverage', action='store_true', help='If set, only get coverage info without cutting patches.')

    args = parser.parse_args()

    if args.command == 'convert':
        convert_sar(args.sar_dir, args.dir_to_save)
    elif args.command == 'process':
        process_sar(args.sar_dir, args.target_dir, args.get_coverage)

if __name__ == '__main__':
    main()
