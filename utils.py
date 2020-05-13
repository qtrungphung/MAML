import matplotlib.pyplot as plt
import os
import shutil
import torch
from data_gen import gen_tasks
import torch
import torch.nn.functional as F 


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

def model_test(model):
    num_samples = 100 

    # Generate tasks
    for data in gen_tasks(1, num_samples):
        x, y, wf = data
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)
    print("loss", loss)
    plt.scatter(x, y, label='Ground Truth')
    plt.scatter(x, y_hat.detach().numpy(), label='Prediction')
    plt.savefig("model_test.png")
    plt.show()

