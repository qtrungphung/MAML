import torch
import torch.nn.functional as F
from utils import model_plot

def adapt_model(model, x, y, K:int=1):
    """Adapt a model to (x,y) set with K GD steps"""

    optim = torch.optim.SGD(model.parameters(), lr=0.1)
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

def main():
    # Hyper params
    num_tasks = {'train': 200,
                 'dev': 10,
                 'test': 1
                 }
    num_samples = {'adapt': 10,
                   'meta': 10
                   }
    tasks = {'train': [],
             'dev': [],
             'test': []
             }
    num_epochs = 100
    alpha = 0.01   # task learning rate
    beta = 0.01    # meta learning rate
    K = 1    # number of GD steps when adapt to a task

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
    model = BasedRegressor()
    untrained_model = copy.deepcopy(model)
    meta_optim = torch.optim.Adam(model.parameters(), lr=beta)

    # --- Training ---
    for epoch in range(num_epochs):
        # --- One meta update step ---
        meta_optim.zero_grad()
        for i in range(num_tasks['train']):
            x = tasks['train'][i]['x']
            y = tasks['train'][i]['y']
            x_meta = tasks['train'][i]['x_meta']
            y_meta = tasks['train'][i]['y_meta']
            fast_weights = list(model.parameters())

            # Adapt to a task
            for k in range(K):
                y_hat = model(x, weight=fast_weights)
                loss = F.mse_loss(y_hat, y)
                # Backward using torch.autograd
                # avoid storing grad in model.parameters() grad
                grads = torch.autograd.grad(loss, model.parameters(),
                                            create_graph=True)
                fast_weights = list(map(lambda a: a[0] - alpha*a[1],
                                        zip(fast_weights, grads)))

            # Meta loss is backprop to model params before task adaptation
            # These gradients are stacked up for all tasks
            y_meta_hat = model(x_meta, weight=fast_weights)
            meta_loss = F.mse_loss(y_meta_hat, y_meta)
            meta_loss.backward()

        # Meta update
        meta_optim.step()

        # --- Validation step ---
        val_loss = torch.zeros(1)
        for i in range(num_tasks['dev']):
            x = tasks['dev'][i]['x']
            y = tasks['dev'][i]['y']
            x_meta = tasks['dev'][i]['x_meta']
            y_meta = tasks['dev'][i]['y_meta']
            val_model = copy.deepcopy(model)
            val_opt = torch.optim.SGD(val_model.parameters(), lr=alpha)

            # forward
            val_opt.zero_grad()
            y_hat = val_model(x)
            loss = F.mse_loss(y_hat, y)
            # backward
            loss.backward()
            # update weights
            with torch.no_grad():
                val_opt.step()

            # Val loss
            y_meta_hat = val_model(x_meta)
            loss = F.mse_loss(y_meta_hat, y_meta)
            val_loss += loss / num_tasks['dev']

        print("epoch {}, val loss {}".format(epoch, val_loss.item()))
    torch.save(model.state_dict(), './updated_model_state_dict.pt')

    # --- Testing ---
    print("--- Testing ---")
    x = tasks['test'][0]['x']
    y = tasks['test'][0]['y']
    x_meta = tasks['test'][0]['x_meta']
    y_meta = tasks['test'][0]['y_meta']
    GD_step = 100

    print("MAML before training, before adaptation")
    model_plot(untrained_model, x, y)
    print("MAML after training, adapted to test task using {}".format(len(x)),
          "data points, {} gradient steps".format(GD_step))
    adapt_untrained_model = adapt_model(untrained_model, x, y, K=GD_step)
    model_plot(adapt_untrained_model, x, y)

    print("MAML after training, before adaptation")
    model_plot(model, x, y)
    print("MAML after training, adapted to test task using {}".format(len(x)),
          "data points, {} gradient steps".format(GD_step))
    adapted_model = adapt_model(model, x, y, K=GD_step)
    model_plot(adapted_model, x, y)

if __name__=="__main__":
    print("Starting")
    main()
    print("Finished")