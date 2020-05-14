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

def reg_test(model):
    """Do a regression prediction test on model"""
    # Generate tasks
    num_samples = 100 
    for data in gen_tasks(1, num_samples):
        x, y, _ = data
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    # Testing
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)
    print("loss", loss)
    plt.scatter(x, y, label='Ground Truth')
    plt.scatter(x, y_hat.detach().numpy(), label='Prediction')
    plt.savefig("model_test.png")
    plt.show()

def reg_test_B(models:list):
    """Do a regression prediction test on model.
    Expect list of model
    """
    # Generate tasks
    num_samples = 100 
    for data in gen_tasks(1, num_samples):
        x, y, _ = data
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    # Testing
    y_hat = torch.zeros(1)
    Z = torch.zeros(1)
    for i in range(len(models)):
        o = models[i](x)
        unnorm_p = torch.exp(-F.mse_loss(o, y))
        y_hat = y_hat + o*unnorm_p
        Z = Z + unnorm_p
    y_hat = y_hat/Z
    loss = F.mse_loss(y_hat, y)
    print("loss", loss)
    plt.scatter(x, y, label='Ground Truth')
    plt.scatter(x, y_hat.detach().numpy(), label='Prediction')
    plt.savefig("model_test.png")
    plt.show()