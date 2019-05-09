from keras.utils import plot_model


def plot_model(model, path):
    plot_model(model, to_file=path)
