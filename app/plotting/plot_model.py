from keras.utils.vis_utils import plot_model


def plotting(model, path):
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
