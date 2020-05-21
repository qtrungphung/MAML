from tqdm import tqdm as tqdm
import numpy as np
import shutil
import os

from config import MINI_IMG_PATH
from utils import mkdir, rmdir

"""code taken from https://github.com/oscarknagg/few-shot.git"""

"""
Run this script to prepare the miniImageNet dataset.
This script uses the 100 classes of 600 images each used in
the Matching Networks paper. The exact images used are
given in data/mini_imagenet.txt which is downloaded from the link
provided in the paper (https://goo.gl/e3orz6).
1. Download files from
https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view
and place in data/miniImageNet/images
2. Run the script
"""


def prepare_mini_imagenet():
    print("preparing dataset")
    # Clean up folders
    rmdir(MINI_IMG_PATH + '/images_background')
    rmdir(MINI_IMG_PATH + '/images_evaluation')
    mkdir(MINI_IMG_PATH + '/images_background')
    mkdir(MINI_IMG_PATH + '/images_evaluation')

    # Find class identities
    classes = []
    for root, _, files in os.walk(MINI_IMG_PATH + '/images/'):
        for f in files:
            if f.endswith('.jpg'):
                classes.append(f[:-12])

    classes = list(set(classes))

    # Train/test split
    np.random.seed(0)
    np.random.shuffle(classes)
    background_classes, evaluation_classes = classes[:80], classes[80:]

    # Create class folders
    for c in background_classes:
        mkdir(MINI_IMG_PATH + f'/images_background/{c}/')

    for c in evaluation_classes:
        mkdir(MINI_IMG_PATH + f'/images_evaluation/{c}/')

    # Move images to correct location
    for root, _, files in os.walk(MINI_IMG_PATH + '/images'):
        for f in tqdm(files, total=600*100):
            if f.endswith('.jpg'):
                class_name = f[:-12]
                image_name = f[-12:]
                # Send to correct folder
                subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
                src = f'{root}/{f}'
                dst = MINI_IMG_PATH + f'/{subset_folder}/{class_name}/{image_name}'
                shutil.copy(src, dst)


if __name__ == "__main__":
    print("preparing mini-imagenet")
    prepare_mini_imagenet()
    print("Done")
