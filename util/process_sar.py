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

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

SAR_DATA_FOLDER = os.getenv('SAR_DATA_FOLDER')
GHSL_DATA_FOLDER = os.getenv('GHSL_DATA_FOLDER')
DATA_FOLDER = os.getenv('DATA_FOLDER')
GHSL = os.getenv('GHSL')
PATCH_FOLDER = os.getenv('PATCH_DATA_FOLDER')

def detect_buildings(ghsl_path, sar, threshold = BUILDINGS_THRESHOLD):
    with rasterio.open(ghsl_path) as ghsl:
        # Get bounds of SAR image in GHSL CRS(WGS84)
        wgs84_bounds = transform_bounds(sar.crs, ghsl.crs, *sar.bounds)

        window = from_bounds(*wgs84_bounds, ghsl.transform)

        ghsl_subset = ghsl.read(1, window=window, out_shape = (sar.height, sar.width))

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

        no_data = ghsl.nodata
        valid_mask = (ghsl_resampled != no_data)
        valid_pixel_count = np.count_nonzero(valid_mask)

        mask = (ghsl_resampled > threshold) & valid_mask
        building_pixel_count = np.count_nonzero(mask)
        
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
    img_width, img_height = img.shape

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
                with rasterio.open(patch_path, 'w', driver='GTiff', width=patch_size, height=patch_size, count=1, dtype=img.dtypes[0], crs=img.crs, transform=img.window_transform(window)) as dst:
                    dst.write(patch, 1)
    gc.collect()    

def main():
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    years = [2023, 2024, 2025]
    ghsl_path = f'{GHSL_DATA_FOLDER}/{GHSL}'
    
    print('Processing SAR images')
    for year in years:
        dir_names = os.listdir(f'{SAR_DATA_FOLDER}/{year}')
        for dir_name in dir_names:
            if 'geo' in dir_name.lower():
                img_path = f'{SAR_DATA_FOLDER}/{year}/{dir_name}/{dir_name}.tif'
                sar = rasterio.open(img_path)
                building_ratio = detect_buildings(ghsl_path, sar, threshold = BUILDINGS_THRESHOLD)
                print(f"Total building coverage for {dir_name}: {building_ratio}")
                if building_ratio >= MIN_BUILDING_COVG:
                    cut_patches(sar, dir_name, f'{DATA_FOLDER}/sar_patches')
                sar.close()
 

if __name__ == '__main__':
    main()
