import copy
import torch
import torch.nn.functional as F
import numpy as np
from data_gen import gen_tasks
from learner import BasedRegressor
# import matplotlib.pyplot as plt


def adapt_model(model, lr, x, y, K: int = 1):
    """Adapt a model to (x,y) set with K GD steps"""

    optim = torch.optim.SGD(model.parameters(), lr=lr)
    for k in range(K):
        # forward
        optim.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        # backward
        loss.backward()
        # update weights
        with torch.no_grad():
            optim.step()
    return model


# def model_plot(model, xs, x, y=None):
#     model.prep_int_approx(x,y)
#     with torch.no_grad():
#         y_hat = model(x, y)
#         y_hat = model.predict(xs)
#     fig = plt.figure(figsize=(5, 4))
#     if y is not None:
#         plt.scatter(x, y, label='Ground Truth')
#     plt.scatter(xs, y_hat, label='Prediction')
#     plt.show()

def test():
    model_stack = []
    n_models = 120
    for i in range(n_models):
        model = BasedRegressor()
        model.load_state_dict(torch.load(
            './updated_BM_state_dict_m{}.pt'.format(i)))
        model_stack.append(model)

    num_tasks = {'train': 1,
                 'dev': 1,
                 'test': 1
                 }
    num_samples = {'adapt': 10,
                   'meta': 1
                   }
    tasks = {'train': [],
             'dev': [],
             'test': []
             }

    # Generate tasks
    for phase in ['train', 'dev', 'test']:
        col_data = gen_tasks(num_tasks[phase],
                             num_samples['adapt'] + num_samples['meta'])
        for i, data in enumerate(col_data):
            task = {'f': None,
                    'x': None,
                    'y': None,
                    'x_meta': None,
                    'y_meta': None
                    }
            X, Y, task['f'] = data
            task['x'] = torch.as_tensor(X[:num_samples['adapt']],
                                        dtype=torch.float32)
            task['y'] = torch.as_tensor(Y[:num_samples['adapt']],
                                        dtype=torch.float32)
            task['x_meta'] = torch.as_tensor(X[num_samples['adapt']:],
                                             dtype=torch.float32)
            task['y_meta'] = torch.as_tensor(Y[num_samples['adapt']:],
                                             dtype=torch.float32)
            tasks[phase].append(task)
    
    x = tasks['train'][0]['x']
    y = tasks['train'][0]['y']

    y_hat = torch.zeros(1)
    Z = torch.zeros(1)

    for i in range(n_models):
        o = model_stack[i](x)
        unnorm_p = torch.exp(-F.mse_loss(o, y))
        y_hat = y_hat + o*unnorm_p
        Z = Z + unnorm_p
    y_hat = y_hat/Z
    loss = F.mse_loss(y_hat, y)
    print("loss", loss)


def main():
    # Hyper params
    num_tasks = {'train': 120,
                 'dev': 10,
                 'test': 1
                 }
    num_samples = {'adapt': 10,
                   'meta': 1
                   }
    tasks = {'train': [],
             'dev': [],
             'test': []
             }
    num_epochs = 1 
    alpha = 0.01  # task learning rate
    beta = 0.01  # meta learning rate
    K = 60  # number of GD steps when adapt to a task

    # Generate tasks
    for phase in ['train', 'dev', 'test']:
        col_data = gen_tasks(num_tasks[phase],
                             num_samples['adapt'] + num_samples['meta'])
        for i, data in enumerate(col_data):
            task = {'f': None,
                    'x': None,
                    'y': None,
                    'x_meta': None,
                    'y_meta': None
                    }
            X, Y, task['f'] = data
            task['x'] = torch.as_tensor(X[:num_samples['adapt']],
                                        dtype=torch.float32)
            task['y'] = torch.as_tensor(Y[:num_samples['adapt']],
                                        dtype=torch.float32)
            task['x_meta'] = torch.as_tensor(X[num_samples['adapt']:],
                                             dtype=torch.float32)
            task['y_meta'] = torch.as_tensor(Y[num_samples['adapt']:],
                                             dtype=torch.float32)
            tasks[phase].append(task)

    # --- Training ---
    model_stack = []
    for epoch in range(num_epochs):
        # --- One meta update step ---
        for i in range(num_tasks['train']):
            model = BasedRegressor()
            optim = torch.optim.Adam(model.parameters(), lr=alpha)
            x = tasks['train'][i]['x']
            y = tasks['train'][i]['y']

            # Adapt to a task
            for step in range(K):
                optim.zero_grad()
                y_hat = model(x)
                loss = F.mse_loss(y_hat, y)
                loss.backward()
                optim.step()
            model_stack.append(model)

    for i in range(len(model_stack)):
        torch.save(model_stack[i].state_dict(),
            './updated_BM_state_dict_m{}.pt'.format(i))
    test()

if __name__ == "__main__":
    print("Starting")
    main()
    print("Finished")

