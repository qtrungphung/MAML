import copy
import torch
import torch.nn.functional as F
from learner import MAMLImageNet
from dataset import MiniImageNet, NShotTaskSampler
from sklearn.preprocessing import LabelEncoder
from config import MODEL_PATH


# Hyper params
num_epochs = 1
alpha = 0.01  # task learning rate
beta = 0.01  # meta learning rate
# 5 shot, 2 tasks, 5 ways
# train GD 5 steps
# meta examples 15
K = 5  # number of GD steps when adapt to a task
n_train = 5
k_train = 5
q_train = 15
num_tasks = 2
task_samples = (n_train + q_train) * k_train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adapt_model(model, lr, x, y, K: int = 1):
    """Adapt a model to (x,y) set with K GD steps"""

    optim = torch.optim.SGD(model.parameters(), lr=lr)
    for k in range(K):
        # forward
        optim.zero_grad()
        y_hat = model(x, None)
        loss = F.mse_loss(y_hat, y)
        # backward
        loss.backward()
        # update weights
        with torch.no_grad():
            optim.step()
    return model


def train():
    # download_miniImageNet()
    # prepare_mini_imagenet()
    background = MiniImageNet('background')
    background_taskloader = torch.utils.data.DataLoader(
        background,
        batch_sampler=NShotTaskSampler(
            dataset=background,
            episodes_per_epoch=60000,
            n_shot=n_train,
            k_way=k_train,
            q_query=q_train,
            num_tasks=num_tasks),
        num_workers=8
    )

    # Model
    model = MAMLImageNet(k_train).to(device)
    torch.save(
        model.state_dict(),
        MODEL_PATH + '/init_mini_' + str(n_train) + 's_' + str(k_train) + 'w.pt'
    )
    meta_optim = torch.optim.Adam(model.parameters(), lr=beta)

    # --- Training ---
    print("Training Start...")
    model.train()
    for epoch in range(num_epochs):
        # 32 batches per epoch
        # each batch comes with num_tasks tasks
        # 1 batch = adapt on num_tasks, calc meta_loss
        # 1 batch = 1 meta-step
        count = 0
        for batch_xs, batch_ys in background_taskloader:
            # --- One meta update step ---
            meta_optim.zero_grad()
            for i_task in range(0, len(batch_xs), task_samples):
                task_xs = batch_xs[i_task:i_task + task_samples]
                task_ys = batch_ys[i_task:i_task + task_samples]
                encoder = LabelEncoder()
                task_ys_en = encoder.fit_transform(task_ys)
                task_ys_en = torch.as_tensor(task_ys_en, dtype=torch.long)

                train_xs = task_xs[:n_train*k_train].to(device)
                train_ys = task_ys_en[:n_train*k_train].to(device)
                meta_xs = task_xs[n_train*k_train:].to(device)
                meta_ys = task_ys_en[n_train*k_train:].to(device)

                fast_weights = dict(model.named_parameters())

                # Adapt to a task
                for k in range(K):
                    y_hat = model(train_xs, params_dict=fast_weights)
                    loss = F.cross_entropy(y_hat, train_ys)
                    # Backward using torch.autograd
                    # avoid storing grad in model.parameters() grad
                    grads = torch.autograd.grad(
                        loss, fast_weights.values(), create_graph=True)
                    fast_weights = dict(
                        (name, param - alpha * grad)
                        for ((name, param), grad)
                        in zip(fast_weights.items(), grads))

                # Meta loss is backprop to model params before task adaptation
                # These gradients are stacked up for all tasks
                y_meta_hat = model(meta_xs, params_dict=fast_weights)
                meta_loss = F.cross_entropy(y_meta_hat, meta_ys)
                meta_loss.backward()

            # Meta update
            meta_optim.step()
            count += 1
            if (count % 5000) == 0:
                torch.save(
                    model.state_dict(),
                    MODEL_PATH + '/model_state_dict_{}.pt'.format(count))
            print("\r ", count, end="")


def test():
    evaluation = MiniImageNet('evaluation')
    evaluation_taskloader = torch.utils.data.DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(
            dataset=evaluation,
            episodes_per_epoch=1,
            n_shot=n_train,
            k_way=k_train,
            q_query=q_train,
            num_tasks=num_tasks),
        num_workers=8
    )
    K = 10

    print("Testing start...")

    model = MAMLImageNet(k_train, 32).to(device)
    model.load_state_dict(torch.load(
        MODEL_PATH + '/model_state_dict_60000.pt'))

    num_epochs = 20
    for epoch in range(num_epochs):
        model.load_state_dict(torch.load(
            MODEL_PATH + '/model_state_dict_60000.pt'))
        for batch_xs, batch_ys in evaluation_taskloader:
            test_model = copy.deepcopy(model)
            opt = torch.optim.SGD(test_model.parameters(), lr=0.01)
            meta_acc = []
            # --- One meta update step ---
            for i_task in range(0, len(batch_xs), task_samples):
                task_xs = batch_xs[i_task:i_task + task_samples]
                task_ys = batch_ys[i_task:i_task + task_samples]
                encoder = LabelEncoder()
                task_ys_en = encoder.fit_transform(task_ys)
                task_ys_en = torch.as_tensor(task_ys_en, dtype=torch.long)

                train_xs = task_xs[:n_train*k_train].to(device)
                train_ys = task_ys_en[:n_train*k_train].to(device)
                meta_xs = task_xs[n_train*k_train:].to(device)
                meta_ys = task_ys_en[n_train*k_train:].to(device)

                # Adapt to a task
                for k in range(K):
                    opt.zero_grad()
                    y_hat = test_model(train_xs)
                    loss = F.cross_entropy(y_hat, train_ys)
                    # Backward using torch.autograd
                    # avoid storing grad in model.parameters() grad
                    loss.backward()
                    opt.step()

                # Meta loss is backprop to model params before task adaptation
                # These gradients are stacked up for all tasks
                y_meta_hat = test_model(meta_xs)
                _, preds = torch.max(y_meta_hat, 1)
                acc = torch.div(
                    torch.sum(preds == meta_ys), float(len(meta_ys)))
                meta_acc.append(acc)
            print("try {}, mean acc {}".format(
                epoch, torch.mean(torch.as_tensor(meta_acc))))


if __name__ == "__main__":
    print("start training...")
    train()
    print("finish training")
