import copy
import torch
import torch.nn.functional as F
import numpy as np
from data_gen import gen_tasks
from learner import BayesRegressor
import matplotlib.pyplot as plt


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


def model_plot(model, xs, x, y=None):
    model.prep_int_approx(x,y)
    with torch.no_grad():
        y_hat = model(x, y)
        y_hat = model.predict(xs)
    fig = plt.figure(figsize=(5, 4))
    if y is not None:
        plt.scatter(x, y, label='Ground Truth')
    plt.scatter(xs, y_hat, label='Prediction')
    plt.show()


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
    num_epochs = 40
    alpha = 0.01  # task learning rate
    beta = 0.01  # meta learning rate
    K = 1  # number of GD steps when adapt to a task

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

    # Model
    model = BayesRegressor()
    torch.save(model.state_dict(), './untrained_Bayes_state_dict.pt')
    untrained_model = copy.deepcopy(model)
    meta_optim = torch.optim.Adam(model.parameters(), lr=beta)

    f = open('./training_log.txt', 'w')

    # --- Training ---
    for epoch in range(num_epochs):
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
    torch.save(model.state_dict(), './updated_Bayes_state_dict.pt')
    f.close()

    # --- Testing ---
    print("--- Testing ---")
    x = tasks['test'][0]['x']
    y = tasks['test'][0]['y']
    x_range = np.arange(-10, 10, 0.001)
    xs = torch.as_tensor(np.random.choice(x_range, size=(100, 1)),
                         dtype=torch.float32)

    print("Bayes before training")
    model_plot(untrained_model, xs, x, y)

    print("Bayes after training")
    model_plot(model, xs, x, y)


if __name__ == "__main__":
    print("Starting")
    main()
    print("Finished")
