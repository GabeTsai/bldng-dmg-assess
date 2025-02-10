import rasterio
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling
from rasterio.windows import Window
from skimage import exposure
import numpy as np
from dotenv import load_dotenv
from config.constants import *
import os
import gc 
import psutil
import logging
import sys

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

SAR_DATA_FOLDER = os.getenv('SAR_DATA_FOLDER')
GHSL_DATA_FOLDER = os.getenv('GHSL_DATA_FOLDER')
DATA_FOLDER = os.getenv('DATA_FOLDER')
GHSL = os.getenv('GHSL')
PATCH_FOLDER = os.getenv('PATCH_DATA_FOLDER')

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sar_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_memory(message):
    """Log memory usage with a custom message"""
    mem = get_memory_usage()
    logging.debug(f"{message} - Memory usage: {mem:.2f} MB")

def detect_buildings(ghsl_path, sar, threshold = BUILDINGS_THRESHOLD):
    log_memory("Starting detect_buildings")
    
    with rasterio.open(ghsl_path) as ghsl:
        # Log image dimensions and data type
        logging.debug(f"SAR dimensions: {sar.width}x{sar.height}, dtype: {sar.dtypes[0]}")
        logging.debug(f"GHSL dimensions: {ghsl.width}x{ghsl.height}, dtype: {ghsl.dtypes[0]}")
        
        wgs84_bounds = transform_bounds(sar.crs, ghsl.crs, *sar.bounds)
        log_memory("After transform_bounds")

        window = from_bounds(*wgs84_bounds, ghsl.transform)
        
        # Log window dimensions
        logging.debug(f"Window dimensions: {window.width}x{window.height}")
        
        ghsl_subset = ghsl.read(1, window=window, out_shape = (sar.height, sar.width))
        log_memory("After reading GHSL subset")

        ghsl_resampled = np.zeros((sar.height, sar.width), dtype = np.uint16)
        log_memory("After creating resampled array")

        reproject(
            source=ghsl_subset,
            destination=ghsl_resampled,
            src_transform=ghsl.window_transform(window),
            src_crs=ghsl.crs,
            dst_transform=sar.transform,    
            dst_crs=sar.crs,
            resampling=Resampling.nearest
        )
        log_memory("After reprojection")

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
        
        log_memory("End of detect_buildings")
        
    return (building_pixel_count / valid_pixel_count) if valid_pixel_count > 0 else 0.0

def contrast_stretch(img, lower_percent=0.1, upper_percent=99.5, gamma = 1.8):
    log_memory("Starting contrast_stretch")
    
    # Log input array info
    logging.debug(f"Input image shape: {img.shape}, dtype: {img.dtype}")
    
    lower, upper = np.percentile(img, (lower_percent, upper_percent))
    img = exposure.rescale_intensity(img, in_range=(lower, upper))
    log_memory("After rescale_intensity")
    
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    log_memory("After CLAHE")
    
    img = exposure.adjust_gamma(img, gamma)
    log_memory("After gamma adjustment")
    
    return img

def cut_patches(img, img_name, dir, patch_size = PATCH_SIZE, max_ratio = MAX_RATIO):
    log_memory(f"Starting cut_patches for {img_name}")
    logging.debug(f"Image dimensions: {img.width}x{img.height}")
    
    img_width, img_height = img.shape
    patch_count = 0
    
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            if patch_count % 100 == 0:  # Log every 100 patches
                log_memory(f"Processing patch {patch_count} at position ({i}, {j})")
            
            window = Window(j, i, patch_size, patch_size)
            patch = img.read(1, window = window)
            
            if patch.shape != (patch_size, patch_size):
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), 
                                     (0, patch_size - patch.shape[1])), 
                             mode='constant', constant_values=0)
            
            if np.sum(patch == 0) / patch.size < max_ratio:
                patch_path = f'{dir}/{img_name}_patch_{i}_{j}.tif'
                patch = contrast_stretch(patch)
                
                with rasterio.open(patch_path, 'w', 
                                 driver='GTiff', 
                                 width=patch_size, 
                                 height=patch_size, 
                                 count=1, 
                                 dtype=img.dtypes[0], 
                                 crs=img.crs, 
                                 transform=img.window_transform(window)) as dst:
                    dst.write(patch, 1)
                
                del patch
            patch_count += 1
            
        # Force garbage collection every row
        gc.collect()
        log_memory(f"Completed row {i}")
    
    log_memory(f"Finished cut_patches for {img_name}")

def main():
    years = [2023, 2024, 2025]
    ghsl_path = f'{GHSL_DATA_FOLDER}/{GHSL}'
    
    logging.info('Processing SAR images')
    for year in years:
        dir_names = os.listdir(f'{SAR_DATA_FOLDER}/{year}')
        for dir_name in dir_names:
            if 'geo' in dir_name.lower():
                img_path = f'{SAR_DATA_FOLDER}/{year}/{dir_name}/{dir_name}.tif'
                logging.info(f"Processing {img_path}")
                log_memory(f"Before opening {img_path}")
                
                sar = rasterio.open(img_path)
                log_memory(f"After opening {img_path}")
                
                building_ratio = detect_buildings(ghsl_path, sar, threshold = BUILDINGS_THRESHOLD)
                logging.info(f"Total building coverage for {dir_name}: {building_ratio}")
                
                if building_ratio >= MIN_BUILDING_COVG:
                    cut_patches(sar, dir_name, f'{DATA_FOLDER}/sar_patches')
                
                sar.close()
                log_memory(f"After closing {img_path}")
                gc.collect()
                log_memory("After garbage collection")

if __name__ == '__main__':
    main()