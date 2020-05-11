from utils import mkdir, rmdir
import os

# Download Mini ImageNet dataset
DATA_PATH = './data/miniImageNet'
file_id = '0B3Irx3uQNoBMQ1FlNXJsZUdYWEE'
download_dest = DATA_PATH + '/images.zip'
extract_dest = DATA_PATH + '/images'

# Model save dir
MODEL_PATH = './model'

# Clean up folders
if not os.path.exists(MODEL_PATH):
    mkdir(MODEL_PATH)
if not os.path.exists('./data'):
    mkdir('./data')
else:
    rmdir(DATA_PATH)
mkdir(DATA_PATH)