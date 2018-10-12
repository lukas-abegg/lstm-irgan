import app.parameters as params
import app.data_preparation.init_data as init
import app.evaluation.evaluate_all as eval
from app.gan.discriminator import Discriminator
from app.gan.generator import Generator

import os
import tensorflow as tf
from keras import backend

from app.training.train import train


def __init_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)

    backend.set_session(sess)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    return backend, sess


def __prepare_data():
    x_data, y_data, documents_data, queries_data = init.get_data()
    feature_size = params.FEATURE_SIZE
    return x_data, y_data, documents_data, queries_data, feature_size


def __train(x_data, y_data, documents_data, queries_data, feature_size, sess, backend) -> (Discriminator, Generator):
    disc_best, gen_best, x_test, y_test = train(x_data, y_data, documents_data, queries_data, feature_size, backend)
    eval.evaluate(gen_best, sess, x_test, y_test)
    return disc_best, gen_best


def __evaluate(generator, sess, dataset):
    eval.evaluate(generator, sess, dataset)


def main(mode):
    backend, sess = __init_config()
    x_data, y_data, documents_data, queries_data, feature_size = __prepare_data()

    if params.TRAIN_MODE == mode:
        discriminator, generator = __train(x_data, y_data, documents_data, queries_data, feature_size, sess, backend)
        discriminator.save_model("/temp/disc")
        generator.save_model("/temp/gen")
    elif params.EVAL_MODE == mode:
        generator = Generator.create_model(feature_size)
        generator.load_from_file("/temp/gen")
        #__evaluate(generator, sess, dataset)
    else:
        print("unknown MODE")


if __name__ == '__main__':
    main(params.TRAIN_MODE)
