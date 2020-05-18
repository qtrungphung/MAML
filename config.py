from utils import mkdir, rmdir
import os

# Download Mini ImageNet dataset
DATA_PATH = './data'
MINI_IMG_PATH = DATA_PATH + '/miniImageNet'
OMNI_PATH = DATA_PATH + '/Omniglot'

file_id = '0B3Irx3uQNoBMQ1FlNXJsZUdYWEE'
download_dest = MINI_IMG_PATH + '/images.zip'
extract_dest = MINI_IMG_PATH + '/images'

# Model save dir
MODEL_PATH = './model'

