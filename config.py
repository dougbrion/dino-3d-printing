ALL_FILES = "/home/cam/Documents/FastData/dino-test/final.csv"
TRAIN_FILES = "/home/cam/Documents/FastData/dino-test/train.csv"
VAL_FILES = "/home/cam/Documents/FastData/dino-test/train.csv"
TEST_FILES = "/home/cam/Documents/FastData/dino-test/train.csv"
TRAIN_DIR = "/home/cam/Documents/FastData/"
IMAGE_SIZE = 256
DATASET_MEAN = [0.4046205, 0.40267563, 0.43739837]
DATASET_STD = [0.1598225, 0.16807246, 0.17369135]
PATCH_SIZE = 16
ZERO_PCT = 0.1
PATCHES_PER_ROW = (IMAGE_SIZE // PATCH_SIZE)
NUM_PATCHES = PATCHES_PER_ROW ** 2
RGB_CHANNELS = 3
NUM_PIXELS = PATCH_SIZE ** 2 * RGB_CHANNELS
VALID_IMAGES = 5
TOPK = 5

# Training parameters
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

# Transformer parameters
N_HEADS = 8
N_LAYERS = 6

# Update constants
TEMPERATURE_S = 0.1
TEMPERATURE_T = 0.05
CENTER_MOMENTUM = 0.9
TEACHER_MOMENTUM = 0.995