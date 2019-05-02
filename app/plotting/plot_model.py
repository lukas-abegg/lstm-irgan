from keras.utils import plot_model
import parameters as params


def plot_model(model):
    plot_model(model, to_file=params.PLOTTED_MODEL_FILE)
