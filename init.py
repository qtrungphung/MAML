"""Init some settings for subsequent run

"""

from utils import rmdir, mkdir
import os
import config

if not os.path.exists(config.DATA_PATH):
    # rmdir(config.DATA_PATH)
    mkdir(config.DATA_PATH)
if not os.path.exists(config.MINI_IMG_PATH):
    # rmdir(config.MINI_IMG_PATH)
    mkdir(config.MINI_IMG_PATH)
if not os.path.exists(config.OMNI_PATH):
    # rmdir(config.OMNI_PATH)
    mkdir(config.OMNI_PATH)
if not os.path.exists(config.MODEL_PATH):
    # rmdir(config.MODEL_PATH)
    mkdir(config.MODEL_PATH)
