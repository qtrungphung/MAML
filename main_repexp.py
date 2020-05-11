import torch
from data_gen import gen_tasks


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
    alpha = 0.01

    # Gen task
    for phase in ['train', 'dev', 'test']:
        col_data = gen_tasks(num_tasks[phase],
                             num_samples['adapt'] + num_samples['meta'])
        for data in col_data:
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
    based_model = BasedRegressor()
    be4_model = copy.deepcopy(based_model)
    model_opt = torch.optim.Adam(based_model.parameters(), lr=0.01)

    fast_weights = list(based_model.parameters())

    for epoch in range(num_epochs):
        # --- Meta update ---

        total_task_loss = 0

        for task in range(num_tasks['train']):
            x = tasks['train'][task]['x']
            y = tasks['train'][task]['y']
            x_meta = tasks['train'][task]['x_meta']
            y_meta = tasks['train'][task]['y_meta']

            # --- find representation ---
            # init R
            rep = Rep()
            fast_rep = list(rep.parameters())
            for k in range(100):
                # forward
                loss = 0
                for j in range(num_samples['adapt']):
                    x_rep = rep(x[j], fast_rep)
                    y_hat = based_model(x_rep)
                    loss += F.mse_loss(y_hat, y[j])
                # backward
                grads = torch.autograd.grad(loss, fast_rep,
                                            create_graph=True)
                # update weights
                fast_rep = list(map(lambda x: x[0] - alpha * x[1],
                                    zip(fast_rep, grads)))

            # new loss
            loss = 0
            for j in range(num_samples['adapt']):
                x_rep = rep(x[j], fast_rep)
                y_hat = based_model(x_rep)
                loss = F.mse_loss(y_hat, y[j])
            total_task_loss += loss
            print('epoch', epoch, 'task', task, 'loss', loss)
        print('epoch', epoch, 'total task loss', total_task_loss)
        total_task_loss.backward()
        model_opt.step()
