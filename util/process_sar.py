import rasterio
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling
from rasterio.windows import Window
from skimage import exposure
import numpy as np
from dotenv import load_dotenv
from config.constants import TILE_WIDTH, MAX_RATIO
import os

load_dotenv('../config.env')
SAR_DATA_FOLDER = os.getenv('SAR_DATA_FOLDER')
GHSL_PREFIX = os.getenv('GHSL_PREFIX')
PATCH_FOLDER = os.getenv('PATCH_DATA_FOLDER')

def find_ghsl_tile(lon, lat):
    
    # GHSL uses a 10x10 degree tile system
    # Calculate the tile coordinates
    
    # Floor division to get tile numbers
    tile_lon = int(lon // TILE_WIDTH) * TILE_WIDTH
    tile_lat = int(lat // TILE_WIDTH) * TILE_WIDTH
    
    # Format according to GHSL naming convention
    # GHSL tiles are named like: GHS_BUILT_S_E<epoch>_GLOBE_R2023A_54009_100_V1_0_R6_C<col>_R<row>
    # where col and row are based on the 5-degree grid
    
    col = int((tile_lon + 180) // TILE_WIDTH)
    row = int((90 - tile_lat) // TILE_WIDTH)
    
    return row, col

def check_tiles(sar):
    bounds = sar.bounds
    wgs84_bounds = transform_bounds(sar.crs, 'EPSG:4326',
                                  bounds.left, bounds.bottom,
                                  bounds.right, bounds.top)

    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    
    # Find which tile each corner belongs to
    corners = [
        (lon_min, lat_min),  # bottom left
        (lon_min, lat_max),  # top left
        (lon_max, lat_min),  # bottom right
        (lon_max, lat_max)   # top right
    ]
    
    # Get unique tile numbers for each corner
    unique_tiles = set()
    for lon, lat in corners:
        unique_tiles.add(find_ghsl_tile(lon, lat))
    
    spans_multiple = len(unique_tiles) > 1
    
    print(f"SAR Image bounds (WGS84): {wgs84_bounds}")
    print(f"Number of tiles needed: {len(unique_tiles)}")
    
    for row, col in unique_tiles:
        print(f"Column: {col}, Row: {row}")
        
    if spans_multiple:
        print("\nRequired tiles:")
        for tile_lon, tile_lat in unique_tiles:
            print(f"Tile covering {tile_lon}°E to {tile_lon+5}°E, {tile_lat}°N to {tile_lat+5}°N")
    
    return spans_multiple, unique_tiles

def make_masks(img):
    masks = []
    # Convert GHSL to WGS84 (EPSG:4326)
    # Higher resolution since we're dealing with a large area
    spans_multiple, tiles = check_tiles(img)
    for row, col in tiles:
        ghsl_path = f'../data/{GHSL_PREFIX}R{row}_C{col}.tif'
        ghsl = rasterio.open(ghsl_path)

        transform_wgs84, width_wgs84, height_wgs84 = calculate_default_transform(
            ghsl.crs,
            'EPSG:4326',
            ghsl.width,
            ghsl.height,
            *ghsl.bounds,
            resolution=(0.001, 0.001)  # ~100m at the equator
        )
        
        ghsl_wgs84 = np.zeros((height_wgs84, width_wgs84), dtype=ghsl.dtypes[0])
        
        # Reproject GHSL to WGS84
        reproject(
            source=ghsl.read(1),
            destination=ghsl_wgs84,
            src_transform=ghsl.transform,
            src_crs=ghsl.crs,
            dst_transform=transform_wgs84,
            dst_crs='EPSG:4326',
            resampling=Resampling.nearest
        )
        
        reproj_ghsl = np.zeros((img.height, img.width), dtype=ghsl.dtypes[0])
        
        # Project from WGS84 to SAR img's crs, using the SAR image's transform
        reproject(
            source=ghsl_wgs84,
            destination=reproj_ghsl,
            src_transform=transform_wgs84,
            src_crs='EPSG:4326',
            dst_transform=img.transform,
            dst_crs=img.crs,
            resampling=Resampling.nearest
        )

        # Create binary mask for settlement areas, exclude nodata
        mask = (reproj_ghsl > 0) & (reproj_ghsl != ghsl.nodata)
        masks.append((reproj_ghsl, mask))
        
        ghsl.close()

    return masks

def check_buildings_present(masks):
    total_percent = 0
    for reproj_ghsl, mask in masks:
        buildings_present = np.any(mask > 0 )
        valid_pixel_count = np.count_nonzero(reproj_ghsl >= 0)
        print("Buildings present in GHSL mask:", buildings_present)
        # Compute percentage of image covered by buildings
        building_coverage = np.sum(mask > 0) / valid_pixel_count 
        print(f"Building coverage: {building_coverage}")
        total_percent += building_coverage
    return total_percent

def contrast_stretch(img, lower_percent=0.1, upper_percent=99.5, gamma = 1.8):
    # Clip pixel values based on lower/upper percentiles (exclude extreme pixel values)
    lower, upper = np.percentile(img, (lower_percent, upper_percent))
    # Apply contrast stretch
    new_img = exposure.rescale_intensity(img, in_range=(lower, upper))
    # Apply CLAHE - tile the image and apply contrast stretching on each tile (local contrast stretching)
    eq_img = exposure.equalize_adapthist(new_img, clip_limit=0.03)
    # Make image a bit darker
    gamma_cor_img = exposure.adjust_gamma(eq_img, gamma)
    return gamma_cor_img

def too_black(patch, max_ratio = MAX_RATIO):
    black_ratio = np.sum(patch == 0) /patch.size
    return black_ratio > max_ratio

def cut_patches(img, img_name, dir, patch_size = 256, max_ratio = MAX_RATIO):
    img_width, img_height = img.shape

    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            window = Window(j, i, patch_size, patch_size)
            patch = img.read(1, window = window)
            if patch.shape != (patch_size, patch_size):
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), 
                                       (0, patch_size - patch.shape[1])), mode='constant', constant_values=0)
                
            # If patch doesn't have too many black pixels, add it to the list
            if np.sum(patch == 0) / patch.size < max_ratio:
                # Save patch to disk
                patch_path = f'{dir}/{img_name}_patch_{i}_{j}.tif'
                patch = contrast_stretch(patch)
                with rasterio.open(patch_path, 'w', driver='GTiff', width=patch_size, height=patch_size, count=1, dtype=img.dtypes[0], crs=img.crs, transform=img.window_transform(window)) as dst:
                    dst.write(patch, 1)

def main():
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    for year in years:
        dir_names = os.listdir(f'{SAR_DATA_FOLDER}/{year}')
        for dir_name in dir_names:
            if 'geo' in dir_name.lower():
                img_path = f'{SAR_DATA_FOLDER}/{year}/{dir_name}/{dir_name}.tif'
                img = rasterio.open(img_path)
                masks = make_masks(img)
                total_percent = check_buildings_present(masks)
                print(f"Total building coverage: {total_percent}")
                cut_patches(img, dir_name, f'{SAR_DATA_FOLDER}/sar_patches')
                img.close()

if __name__ == '__main__':
    main()