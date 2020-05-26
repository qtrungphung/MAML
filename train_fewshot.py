import os
import argparse
import dataset
import learner
import torch
import torch.nn.functional as F
from config import MODEL_PATH
from sklearn.preprocessing import LabelEncoder


def main():
    # Parsing arguments from terminal
    # An epoch = list of episodes
    # An episode = A meta_batch
    # A batch = [task_1, task_2, ..., task_meta_batch_sz]
    # A task  = support set + query set
    # Support set = n_way*k_shot
    # Query set = n_way*q_query

    # Specific settings:

    # Omniglot
    # 5_ways
    #     inner-lr = 0.4
    #     inner-train-steps = 1, inner-val-steps = 3
    #     meta-batch-sz = 32
    # 20_ways
    #     inner-lr = 0.1
    #     inner-train-steps = 5, inner-val-steps = 5
    #     meta-batch-sz = 16

    # MiniImageNet
    # both 5_way 1_shot and 5_way 5_shot
    #     inner-lr = 0.01
    #     inner-train-steps = 5, inner-val-steps = 10
    # 1-shot
    #     meta-batch-sz = 4
    # 5-shot
    #     meta-batch-sz = 2

    # General settings
    #     q_query = 15
    #     meta-lr = 0.01
    #     epochs: 60000
    #     epoch-len = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--dataset', default='miniimagenet', type=str)
    parser.add_argument('--n-way', default=5, type=int)
    parser.add_argument('--k-shot', default=5, type=int)
    parser.add_argument('--q-query', default=15, type=int)
    parser.add_argument('--inner-lr', default=0.01, type=float)
    parser.add_argument('--inner-train-steps', default=5, type=int)
    parser.add_argument('--inner-val-steps', default=10, type=int)
    parser.add_argument('--meta-lr', default=0.01, type=float)
    # tasks per batch
    parser.add_argument('--meta-batch-sz', default=2, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    # num batches per epoch
    parser.add_argument('--epoch-len', default=1000, type=int)
    parser.add_argument('--save-name', default='fs', type=str)
    args = parser.parse_args()

    if args.dataset == 'omniglot':
        dataset_class = dataset.OmniglotDataset
        n_filters = 64
        C_in = 1
        H_in = 26
    elif args.dataset == 'miniimagenet':
        dataset_class = dataset.MiniImageNet
        n_filters = 32
        C_in = 3
        H_in = 84
    else:
        raise(ValueError('Unsupported dataset'))

    param_str = f'mode={args.mode}_dataset={args.dataset}_' \
                f'n_way={args.n_way}_k_shot={args.k_shot}_' \
                f'q_query={args.q_query}_inner_lr={args.inner_lr}_' \
                f'inner_train_steps={args.inner_train_steps}_' \
                f'inner_val_steps={args.inner_val_steps}_' \
                f'meta_lr={args.meta_lr}_meta_batch_sz={args.meta_batch_sz}_' \
                f'epochs={args.epochs}_epoch_len={args.epoch_len}_' \
                f'save_name={args.save_name}'
    print(param_str)

    # check save path
    if not os.path.isdir(MODEL_PATH):
        print("Model save path is not a directory. Check config")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = learner.MAMLConv(
        n_way=args.n_way,
        n_filters=n_filters,
        C_in=C_in,
        H_in=H_in
    ).to(device)

    if args.mode == 'train':
        data = dataset_class('background')
        torch.save(
            model.state_dict(),
            MODEL_PATH + '/init_' + args.save_name + '.pt'
        )
    elif args.mode == 'test':
        data = dataset_class('evaluation')
        state_dict_file = input("Input state dict file path: ")
        model.load_state_dict(torch.load(state_dict_file))
    else:
        raise(ValueError('Mode must be train or test'))
    model.train()

    taskloader = torch.utils.data.DataLoader(
        data,
        batch_sampler=dataset.NShotTaskSampler(
            dataset=data,
            episodes_per_epoch=args.epoch_len,
            n_shot=args.k_shot,
            k_way=args.n_way,
            q_query=args.q_query,
            num_tasks=args.meta_batch_sz),
        num_workers=8
    )
    meta_optim = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    sup_len = args.k_shot * args.n_way
    que_len = args.q_query * args.n_way
    task_len = sup_len + que_len

    # --- Training ---
    print("{} is starting...".format(args.mode))
    total_acc = []
    for epoch in range(args.epochs):
        # An epoch = list of episodes
        # An episode = A meta_batch
        # A meta_batch = [task_1, task_2, ..., task_meta_batch_sz]
        # A task  = support set + query set
        # Support set = n_way*k_shot
        # Query set = n_way*q_query
        epoch_acc = []

        for meta_batch_X, meta_batch_Y in taskloader:
            # --- One meta update step ---
            # For each task in meta batch, Finetune, then backprop
            meta_optim.zero_grad()
            for idx in range(0, len(meta_batch_X), task_len):
                # A task
                task_X = meta_batch_X[idx:idx + task_len]
                task_Y = meta_batch_Y[idx:idx + task_len]
                encoder = LabelEncoder()
                task_Y = encoder.fit_transform(task_Y)
                task_Y = torch.as_tensor(task_Y, dtype=torch.long)

                sup_X = task_X[:sup_len].to(device)
                sup_Y = task_Y[:sup_len].to(device)
                que_X = task_X[sup_len:].to(device)
                que_Y = task_Y[sup_len:].to(device)

                fast_weights = dict(model.named_parameters())

                # Fine tuning
                for k in range(args.inner_train_steps):
                    y_hat = model(sup_X, params_dict=fast_weights)
                    loss = F.cross_entropy(y_hat, sup_Y)
                    # Backward using torch.autograd
                    # avoid storing grad in model.parameters() grad
                    grads = torch.autograd.grad(
                        loss, fast_weights.values(), create_graph=True)
                    fast_weights = dict(
                        (name, param - args.inner_lr * grad)
                        for ((name, param), grad)
                        in zip(fast_weights.items(), grads))

                y_meta_hat = model(que_X, params_dict=fast_weights)
                if args.mode == 'train':
                    # Meta loss is backprop to model params before fine tuning
                    # These gradients are summed up for a meta-batch
                    meta_loss = F.cross_entropy(y_meta_hat, que_Y)
                    meta_loss.backward()
                else:
                    # Accuracy after fine tuning
                    _, preds = torch.max(y_meta_hat, 1)
                    task_acc = torch.div(
                        torch.sum(preds == que_Y), float(len(que_Y)))
                    epoch_acc.append(task_acc)
            # Meta update
            if args.mode == 'train':
                meta_optim.step()

        # Epoch output
        print("Epoch:", epoch)
        if args.mode == 'train':
            if ((epoch + 1) % int(args.epochs/10)) == 0:
                torch.save(
                    model.state_dict(),
                    MODEL_PATH + '/' + args.save_name + '_{}.pt'.format(epoch)
                )
        else:
            epoch_acc = torch.mean(torch.as_tensor(epoch_acc))
            print("Epoch mean acc {}".format(epoch_acc))
            total_acc.append(epoch_acc)
    # final model
    if args.mode == 'train':
        torch.save(
            model.state_dict(),
            MODEL_PATH + '/' + args.save_name + '_{}_final.pt'.format(epoch)
        )
    else:
        print("Test mean acc: {}".format(
            torch.mean(torch.as_tensor(total_acc))))


if __name__ == '__main__':
    print('Starting...')
    main()
    print('Finished')
