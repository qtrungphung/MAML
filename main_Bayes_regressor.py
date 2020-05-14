import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from data_gen import gen_tasks
from learner import BayesRegressor


# Hyper params
num_tasks = {'train': 120,
             'dev': 10
             }
num_samples = 10
tasks = {'train': [],
         'dev': [],
         }
train_epochs = 40
alpha = 0.01  # task learning rate
beta = 0.01  # meta learning rate
K = 1  # number of GD steps when adapt to a task


def train():
    # Generate tasks
    for phase in ['train', 'dev']:
        col_data = gen_tasks(num_tasks[phase], num_samples)
        for i, data in enumerate(col_data):
            task = {'f': None,
                    'x': None,
                    'y': None,
                    }
            X, Y, task['f'] = data
            task['x'] = torch.as_tensor(X, dtype=torch.float32)
            task['y'] = torch.as_tensor(Y, dtype=torch.float32)
            tasks[phase].append(task)

    # Model
    model = BayesRegressor()
    torch.save(model.state_dict(), './untrained_Bayes_state_dict.pt')
    meta_optim = torch.optim.Adam(model.parameters(), lr=beta)

    f = open('./training_log.txt', 'w')

    # --- Training ---
    for epoch in range(train_epochs):
        # --- One meta update step ---
        model.train()
        meta_optim.zero_grad()
        for i in range(num_tasks['train']):
            x = tasks['train'][i]['x']
            y = tasks['train'][i]['y']

            # Adapt to a task
            # find theta that approx integral
            model.prep_int_approx(x, y)
            # forward
            y_hat = model(x, y)
            loss = F.mse_loss(y_hat, y)
            loss.backward()

        # Meta update
        meta_optim.step()

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_loss = torch.zeros(1)
            for i in range(num_tasks['dev']):
                x = tasks['dev'][i]['x']
                y = tasks['dev'][i]['y']

                # find theta that approx integral
                model.prep_int_approx(x, y)

                # Val loss
                y_hat = model.predict(x)
                loss = F.mse_loss(y_hat, y)
                val_loss += loss / num_tasks['dev']

        print("epoch {}, val loss {}".format(epoch, val_loss.item()))
        f.write("epoch {}, val loss {}".format(epoch, val_loss.item()))
        if (epoch % 5) == 4:
            torch.save(model.state_dict(),
                './updated_Bayes_reg_{}.pt'.format(epoch))
    f.close()


def test():
    model = BayesRegressor()
    model.load_state_dict(torch.load(
        './updated_Bayes_reg_{}.pt'.format(train_epochs-1)))
    # Generate tasks
    num_samples = 100 
    for data in gen_tasks(1, num_samples):
        x, y, wf = data
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    # Testing
    model.prep_int_approx(x, y)
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)
    print("loss", loss)
    plt.scatter(x, y, label='Ground Truth')
    plt.scatter(x, y_hat.detach().numpy(), label='Prediction')
    plt.savefig("model_test.png")
    plt.show()


if __name__ == "__main__":
    print("Starting")
    train()
    test()
    print("Finished")
