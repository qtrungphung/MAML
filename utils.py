import matplotlib.pyplot as plt
import os
import shutil
import torch


def model_plot(model, xs, x, y=None):
    with torch.no_grad():
        y_hat = model(xs)
    fig = plt.figure(figsize=(5, 4))
    if y is not None:
        plt.scatter(x, y, label='Ground Truth')
    plt.scatter(xs, y_hat, label='Prediction')
    plt.show()

def mkdir(dir):
    """Create a directory, ignoring exceptions
        # Arguments:
        dir: Path of directory to create
        https://github.com/oscarknagg/few-shot.git
    """
    try:
        os.mkdir(dir)
    except:
        pass

def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions
        # Arguments:
        dir: Path of directory to recursively remove
        https://github.com/oscarknagg/few-shot.git
    """
    try:
        shutil.rmtree(dir)
    except:
        pass