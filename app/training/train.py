from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import random

import app.parameters as params
from app.gan.generator import Generator
from app.gan.discriminator import Discriminator
import app.evaluation.metrics.precision_k as p_k
import app.evaluation.metrics.ndcg_k as ndcg_k


def train(query_ids, ratings_data, queries_data, documents_data, feature_size, sess):
    x_train, x_test = train_test_split(query_ids, test_size=0.33, random_state=42)

    # Initialize data for eval
    p_best_val = 0.0
    ndcg_best_val = 0.0

    best_disc = Discriminator
    best_gen = Generator

    skf = KFold(n_splits=params.KFOLD_SPLITS, shuffle=True)

    # Loop through the indices the split() method returns
    for index, (train_k_indices, val_k_indices) in enumerate(skf.split(x_train)):
        # Generate batches from indices
        x_train_k, x_val_k = np.array(x_train)[train_k_indices], np.array(x_train)[val_k_indices]

        # Clear model, and create it
        disc = Discriminator.create_model(feature_size)
        gen = Generator.create_model(feature_size)

        gen, disc, p_val, ndcg_val = __train_model(disc, gen, x_train_k, x_val_k, ratings_data, queries_data, documents_data, sess)

        best_disc, best_gen, p_best_val, ndcg_best_val = __get_best_eval_result(disc, best_disc, gen, best_gen, p_val,
                                                                                p_best_val, ndcg_val, ndcg_best_val)

    return best_disc, best_gen, x_test


def __train_model(disc, gen, x_train, x_val, ratings_data, queries_data, documents_data, sess) -> (Discriminator, Generator):
    train_ratings_data, train_queries_data, train_documents_data = __build_train_data(x_train, ratings_data, queries_data, documents_data)

    # Initialize data for eval
    p_best_val = 0.0
    ndcg_best_val = 0.0

    best_disc = Discriminator
    best_gen = Generator

    print('Start adversarial training')
    for epoch in range(params.DISC_TRAIN_EPOCHS):

        # Train Discriminator
        print('Training Discriminator ...')
        for d_epoch in range(params.DISC_TRAIN_GEN_EPOCHS):
            print('now_ D_epoch : ', str(d_epoch))

            pos_neg_data = []
            pos_neg_size = 0
            if d_epoch % params.DISC_TRAIN_EPOCHS == 0:
                # Generator generate negative for Discriminator, then train Discriminator
                pos_neg_data = __generate_negatives_for_discriminator(gen, x_train, train_ratings_data, documents_data)
                pos_neg_size = len(pos_neg_data)

            i = 1
            while i <= pos_neg_size:
                batch_index = i - 1
                if i + params.DISC_BATCH_SIZE <= pos_neg_size:
                    input_pos, input_neg = __get_batch_data(pos_neg_data, batch_index, params.DISC_BATCH_SIZE)
                else:
                    input_pos, input_neg = __get_batch_data(pos_neg_data, batch_index, pos_neg_size - batch_index)

                i += params.DISC_BATCH_SIZE

                # prepare pos and neg data
                pos_data = [[queries_data[x[0]], documents_data[x[1]]] for x in input_pos]
                neg_data = [[queries_data[x[0]], documents_data[x[1]]] for x in input_neg]

                pred_data = []
                pred_data.extend(pos_data)
                pred_data.extend(neg_data)
                pred_data = np.asarray(pred_data)

                # prepara pos and neg label
                pred_data_label = [1.0] * len(pos_data)
                pred_data_label.extend([0.0] * len(neg_data))
                pred_data_label = np.asarray(pred_data_label)

                # train
                disc.train(pred_data, pred_data_label)

        # Train Generator
        print('Training Generator ...')
        for g_epoch in range(params.GEN_TRAIN_EPOCHS):
            print('now_ G_epoch : ', str(g_epoch))

            for query_id in queries_data.index.values:

                # get query specific rating and all relevant docs
                x_pos_list, y_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)
                x_pos_set = set(x_pos_list)

                # Importance Sampling
                prob = gen.get_prob(candidate_list)
                prob = prob[0]
                prob = prob.reshape([-1])

                # important sampling, change doc prob
                prob_IS = prob * (1.0 - params.GEN_LAMBDA)

                for i in range(len(candidate_list)):
                    if candidate_list[i] in x_pos_set:
                        prob_IS[i] += (params.GEN_LAMBDA / (1.0 * len(x_pos_list)))

                # G generate some url (5 * postive doc num)
                choose_index = np.random.choice(np.arange(len(candidate_list)), [5 * len(x_pos_list)], p=prob_IS)
                # choose data
                choose_data = np.array(candidate_list)[choose_index]

                # prob / important sampling prob (loss => prob * reward * prob / important sampling prob)
                choose_IS = np.array(prob)[choose_index] / np.array(prob_IS)[choose_index]

                choose_index = np.asarray(choose_index)
                choose_data = np.asarray(choose_data)
                choose_IS = np.asarray(choose_IS)

                # get reward((prob  - 0.5) * 2 )
                choose_reward = disc.get_preresult(choose_data)

                # train
                gen.train(choose_data[np.newaxis, :], choose_reward.reshape([-1])[np.newaxis, :],
                          choose_IS[np.newaxis, :])

            # Evaluate
            p_step = p_k.measure_precision_at_k(gen, x_val, ratings_data, queries_data, documents_data, params.EVAL_K, sess)
            ndcg_step = ndcg_k.measure_ndcg_at_k(gen, x_val, ratings_data, queries_data, documents_data, params.EVAL_K, sess)

            best_disc, best_gen, p_best_val, ndcg_best_val = __get_best_eval_result(disc, best_disc, gen, best_gen, p_step,
                                                                                    p_best_val, ndcg_step, ndcg_best_val)

    print("Best:", "gen p@5 ", p_best_val, "gen ndcg@5 ", ndcg_best_val)

    return best_disc, best_gen, p_best_val, ndcg_best_val


def __build_train_data(x_train, ratings_data, queries_data, documents_data):
    train_queries_data = {}
    train_documents_data = {}
    train_ratings_data = {}

    for query_id in x_train:
        train_ratings_data[query_id] = ratings_data[query_id]
        train_queries_data[query_id] = queries_data[query_id]
        for key in ratings_data.keys():
            train_documents_data[key] = documents_data[key]

    return train_ratings_data, train_queries_data, train_documents_data


def __generate_negatives_for_discriminator(gen, x_train, ratings_data, documents_data):
    data = []

    print('negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, y_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        # Importance Sampling
        # prob = gen.get_prob(candidate_list)
        # prob = prob[0]
        # prob = prob.reshape([-1])
        #
        # neg_list = np.random.choice(candidate_list, size=[len(x_pos_list)], p=prob)
        neg_list = np.random.choice(candidate_list, size=[len(x_pos_list)])

        for i in range(len(x_pos_list)):
            data.append([query_id, x_pos_list[i], neg_list[i]])

    # shuffle
    random.shuffle(data)
    return data


def __get_query_specific_data(query_id, ratings_data, documents_data):
    # get all query specific ratings
    x_pos_list = list(ratings_data[query_id].keys())
    y_pos_list = list(ratings_data[query_id].values())

    # get all other ratings
    docs_pos_ids = np.unique(x_pos_list)
    candidate_list = []
    for doc_id in documents_data.keys():
        if doc_id not in docs_pos_ids:
            candidate_list.append(doc_id)

    return x_pos_list, y_pos_list, candidate_list


def __get_batch_data(pos_neg_data, index, size):
    pos = []
    neg = []
    for i in range(index, index + size):
        line = pos_neg_data[i]
        pos.append([line[0], line[1]])
        neg.append([line[0], line[2]])
    return pos, neg


def __get_best_eval_result(disc, best_disc, gen, best_gen, p_5, p_best_val, ndcg_5, ndcg_best_val):
    if p_5 > p_best_val:
        p_best_val = p_5
        ndcg_best_val = ndcg_5

        best_gen = gen
        best_disc = disc

    elif p_5 == p_best_val:
        if ndcg_5 > ndcg_best_val:
            ndcg_best_val = ndcg_5

            best_gen = gen
            best_disc = disc

    return best_disc, best_gen, p_best_val, ndcg_best_val
