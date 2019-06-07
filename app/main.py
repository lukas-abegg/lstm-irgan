import os
import warnings

from comet_ml import Experiment

import tensorflow as tf
from keras import backend

import data_preparation.init_data_example as init_example
import data_preparation.init_data_wikiclir as init_wikiclir
import data_preparation.init_data_nfcorpus as init_nfcorpus
import data_preparation.init_data_trec as init_trec
import data_preparation.init_data_trec_ltr as init_trec_ltr
import evaluation.eval_all_metrics as eval_metrics
import evaluation.eval_ltr as eval_ltr
import evaluation.eval_ltr_p5 as eval_ltr_p5
import parameters as params
import plotting.plot_model as plotting
from gan.adverserial_nn.discriminator import Discriminator
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
    elif params.DATA_SOURCE == params.DATA_SOURCE_TREC_CDS_2017:
        print("Init TREC CDS 2017")
        query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = init_trec.get_data()
    elif params.DATA_SOURCE == params.DATA_SOURCE_TREC_CDS_2017_LTR:
        print("Init TREC CDS 2017 LTR")
        query_ids, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = init_trec_ltr.get_data()
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
    learning_rate_d = params.LEARNING_RATE_D
    learning_rate_g = params.LEARNING_RATE_G
    temperature = params.TEMPERATURE
    dropout = params.DROPOUT

    best_gen, best_disc, validation_acc, validation_ndcg = train.train_model(x_train, ratings_data, queries_data, documents_data, tokenizer_q,
                                                 tokenizer_d, sess, weight_decay, learning_rate_d, learning_rate_g, temperature, dropout, experiment)

    return best_gen, best_disc


def evaluate(model, x_data, ratings_data, queries_data, documents_data, sess, experiment: Experiment):
    eval_metrics.evaluate(model, x_data, ratings_data, queries_data, documents_data, sess, experiment)


def evaluate_ltr(model, x_data, ratings_data, queries_data, documents_data, sess, experiment: Experiment):
    eval_ltr.evaluate(model, x_data, ratings_data, queries_data, documents_data, sess, experiment)


def evaluate_ltr_p5(model, x_data, ratings_data, queries_data, documents_data, sess, experiment: Experiment):
    eval_ltr_p5.evaluate(model, x_data, ratings_data, queries_data, documents_data, sess, experiment)


def save_model_to_file(model, path):
    model.save_model_to_file(path)


def save_model_to_weights(model, path_json, path_weights):
    model.save_model_to_weights(path_json, path_weights)


def load_model_from_file(model_class, path):
    return model_class.load_model_from_file(path)


def load_model_from_weights(model_class, path_json, path_weights):
    return model_class.load_model_from_weights(path_json, path_weights)


def plot_model(disc, exp):
    path = params.PLOTTED_MODEL_FILE
    plotting.plotting(disc, path)
    exp.log_image(file_name="Model_TREC2017", file_path=path)


def main(mode, experiment: Experiment):
    if params.TRAIN_MODE == mode:

        sess, x_train, x_val, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = get_env_data_with_x_data_splitted()
        generator, discriminator = train_model_without_hyperparam_opt(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, experiment)

        if generator is not None:
            save_model_to_file(generator, params.SAVED_MODEL_GEN_FILE)
        if discriminator is not None:
            save_model_to_file(discriminator, params.SAVED_MODEL_DISC_FILE)

    elif params.EVAL_MODE == mode:
        sess, x_data, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = get_env_data_not_splitted()
        disc = Discriminator
        disc = load_model_from_file(disc, params.SAVED_MODEL_DISC_FILE)
        evaluate(disc, x_data, ratings_data, queries_data, documents_data, sess, experiment)

    elif params.EVAL_LTR_MODE == mode:
        sess, x_data, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = get_env_data_not_splitted()
        disc = Discriminator
        disc = load_model_from_file(disc, params.SAVED_MODEL_DISC_FILE)
        evaluate_ltr_p5(disc, x_data, ratings_data, queries_data, documents_data, sess, experiment)

    elif params.PLOT_MODEL_MODE == mode:
        disc = Discriminator
        disc = load_model_from_file(disc, params.SAVED_MODEL_DISC_FILE)
        plot_model(disc, experiment)

    else:
        print("unknown MODE")


if __name__ == '__main__':
    experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
                            project_name="ltr-evals", workspace="abeggluk")
    main(params.USED_MODE, experiment)
