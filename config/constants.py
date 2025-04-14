TILE_WIDTH = 10 # 10 degrees tile width for GHSL
MAX_RATIO = 0.75 # Maximum ratio of occurence of one color for patch to be considered (don't want pure black/white patches)
BUILDINGS_THRESHOLD = 100 # GHSL scores settlement between 0 and 100000. We consider a pixel to be a building if it has a score greater than 100
MIN_BUILDING_COVG = 0.30
PATCH_SIZE = 512

FMOW_MEAN = [0.42601296, 0.42821994, 0.40039617]
FMOW_STD = [0.26452429, 0.2589558, 0.26610789]

SAR_MEAN = [2.48371019]
SAR_STD = [0.59492676]
