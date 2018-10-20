import app.parameters as params
import app.data_preparation.init_data as init
import app.evaluation.eval_all_metrics as eval_metrics
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
    query_ids, ratings_data, documents_data, queries_data = init.get_data()
    feature_size = params.FEATURE_SIZE
    return query_ids, ratings_data, documents_data, queries_data, feature_size


def __train(query_ids, ratings_data, queries_data, documents_data, feature_size, sess) -> (Discriminator, Generator):
    best_disc, best_gen, x_test = train(query_ids, ratings_data, queries_data, documents_data, feature_size, sess)
    eval_metrics.evaluate(best_gen, x_test, ratings_data, documents_data, queries_data, sess)
    return best_disc, best_gen


def __evaluate(gen, x_data, ratings_data, queries_data, documents_data, sess):
    eval_metrics.evaluate(gen, x_data, ratings_data, queries_data, documents_data, sess)


def main(mode):
    backend, sess = __init_config()
    query_ids, ratings_data, documents_data, queries_data, feature_size = __prepare_data()

    if params.TRAIN_MODE == mode:
        discriminator, generator = __train(query_ids, ratings_data, queries_data, documents_data, feature_size, sess)
        discriminator.save_model("/temp/disc")
        generator.save_model("/temp/gen")
    elif params.EVAL_MODE == mode:
        generator = Generator.create_model(feature_size)
        generator.load_from_file("/temp/gen")
        __evaluate(generator, query_ids, ratings_data, queries_data, documents_data, sess)
    else:
        print("unknown MODE")


if __name__ == '__main__':
    main(params.TRAIN_MODE)
