import matplotlib.pyplot as plt


def model_plot(model, xs, x, y=None):
    with torch.no_grad():
        y_hat = model(xs)
    fig = plt.figure(figsize=(5, 4))
    if y is not None:
        plt.scatter(x, y, label='Ground Truth')
    plt.scatter(xs, y_hat, label='Prediction')
    plt.show()