"""Init some settings for subsequent run

"""

from utils import rmdir, mkdir
import config

rmdir(config.DATA_PATH)
mkdir(config.DATA_PATH)
rmdir(config.MINI_IMG_PATH)
mkdir(config.MINI_IMG_PATH)
rmdir(config.OMNI_PATH)
mkdir(config.OMNI_PATH)

rmdir(config.MODEL_PATH)
mkdir(config.MODEL_PATH)