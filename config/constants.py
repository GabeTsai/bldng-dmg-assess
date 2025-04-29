TILE_WIDTH = 10 # 10 degrees tile width for GHSL
MAX_RATIO = 0.75 # Maximum ratio of occurence of one color for patch to be considered (don't want pure black/white patches)
BUILDINGS_THRESHOLD = 0 # GHSL scores settlement between 0 and 10000. This score is a rough approximation for enough human settlement in a 100m x 100m patch.
MIN_BUILDING_COVG = 0.30
PATCH_SIZE = 512

FMOW_MEAN = [0.42601296, 0.42821994, 0.40039617]
FMOW_STD = [0.26452429, 0.2589558, 0.26610789]

SAR_MEAN = [2.48371019]
SAR_STD = [0.59492676]
SAR_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
SAR_IMAGE_METADATA_HEADERS = ['imageName', 'latCenter', 'lonCenter']
SAR_IMAGE_BLDNG_COVERAGE_HEADERS = ['imageName', 'buildingCoverage']
PATCH_METADATA_FILENAME = 'image-metadata.csv'
PATCH_BLDNG_COVERAGE_FILENAME = 'building-coverage.csv'