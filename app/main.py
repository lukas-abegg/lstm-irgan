import os
import warnings

from comet_ml import Experiment

import tensorflow as tf
from keras import backend

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform

import data_preparation.init_data_example as init_example
import data_preparation.init_data_wikiclir as init_wikiclir
import data_preparation.init_data_nfcorpus as init_nfcorpus
import evaluation.eval_all_metrics as eval_metrics
import parameters as params
import plotting.plot_model as plotting
from gan.adverserial_nn.generator import Generator
from training import train

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


def __init_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)

    backend.set_session(sess)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    return sess


def __prepare_data():
    if params.DATA_SOURCE == params.DATA_SOURCE_WIKICLIR:
        print("Init WikiClir")
        query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = init_wikiclir.get_data()
    elif params.DATA_SOURCE == params.DATA_SOURCE_NFCORPUS:
        print("Init NFCorpus")
        query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = init_nfcorpus.get_data()
    else:
        print("Init Example")
        query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = init_example.get_data()

    return query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d


def get_env_data_with_x_data_splitted():
    sess = __init_config()
    query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = __prepare_data()
    x_train, x_test = train.get_x_data_splitted(query_ids)
    return sess, x_train, x_test, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d


def get_env_data_not_splitted():
    sess = __init_config()
    query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = __prepare_data()
    return sess, query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d


def train_model_without_hyperparam_opt(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, experiment):
    weight_decay = params.WEIGHT_DECAY
    learning_rate = params.LEARNING_RATE
    temperature = params.TEMPERATURE
    dropout = params.DROPOUT

    best_gen, best_disc, validation_acc, validation_ndcg = train.train_model(x_train, ratings_data, queries_data, documents_data, tokenizer_q,
                                                 tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout, experiment)

    return best_gen, best_disc


def train_model_with_hyperparam_opt(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess):
    weight_decay = params.WEIGHT_DECAY
    learning_rate = {{uniform(params.OPT_MIN_LEARNING_RATE, params.OPT_MAX_LEARNING_RATE)}}
    temperature = {{uniform(params.OPT_MIN_TEMPERATURE, params.OPT_MAX_TEMPERATURE)}}
    dropout = {{uniform(params.OPT_MIN_DROPOUT, params.OPT_MAX_DROPOUT)}}

    best_gen, best_disc, validation_acc, validation_ndcg = train.train_model(x_train, ratings_data, queries_data, documents_data, tokenizer_q,
                                                 tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout)

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': best_gen}


def evaluate(gen, x_data, ratings_data, queries_data, documents_data, sess):
    eval_metrics.evaluate(gen, x_data, ratings_data, queries_data, documents_data, sess)


def save_model_to_file(model, path):
    model.save_model_to_file(path)


def save_model_to_weights(model, path_json, path_weights):
    model.save_model_to_weights(path_json, path_weights)


def load_model_from_file(model_class, path):
    return model_class.load_model_from_file(path)


def load_model_from_weights(model_class, path_json, path_weights):
    return model_class.load_model_from_weights(path_json, path_weights)


def plot_model(gen):
    plotting.plot_model(gen)


def main(mode, experiment):
    if params.TRAIN_MODE == mode:
        if not params.USE_HYPERPARAM_OPT:
            sess, x_train, x_val, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = get_env_data_with_x_data_splitted()
            generator, discriminator = train_model_without_hyperparam_opt(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, experiment)

        else:
            best_run, best_model = optim.minimize(model=train_model_with_hyperparam_opt,
                                                  data=get_env_data_with_x_data_splitted,
                                                  algo=tpe.suggest,
                                                  max_evals=5,
                                                  trials=Trials())

            sess, x_train, x_val, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = get_env_data_with_x_data_splitted()
            generator = best_model
            discriminator = None
            print("Best performing model chosen hyper-parameters:")
            print(best_run)
            eval_metrics.evaluate(generator, x_val, ratings_data, documents_data, queries_data, sess)

        save_model_to_weights(generator, params.SAVED_MODEL_GEN_JSON, params.SAVED_MODEL_GEN_WEIGHTS)
        save_model_to_file(generator, params.SAVED_MODEL_GEN_FILE)
        if discriminator is not None:
            save_model_to_file(discriminator, params.SAVED_MODEL_DISC_FILE)

    elif params.EVAL_MODE == mode:
        sess, x_data, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = get_env_data_not_splitted()
        generator = Generator
        generator = load_model_from_file(generator, params.SAVED_MODEL_GEN_FILE)
        evaluate(generator, x_data, ratings_data, queries_data, documents_data, sess)

    elif params.PLOT_MODEL_MODE == mode:
        generator = Generator
        generator = load_model_from_file(generator, params.SAVED_MODEL_GEN_FILE)
        plot_model(generator)

    else:
        print("unknown MODE")


if __name__ == '__main__':
    experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                            project_name="general", workspace="abeggluk")

    main(params.USED_MODE, experiment)
