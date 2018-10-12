from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import random

import app.parameters as params
from app.gan.generator import Generator
from app.gan.discriminator import Discriminator


def train(x_data, y_data, documents_data, queries_data, feature_size, backend):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

    disc_best = Discriminator
    gen_best = Generator
    acc_best = 0.0

    skf = KFold(n_splits=params.KFOLD_SPLITS, shuffle=True)
    # Loop through the indices the split() method returns
    for index, (train_k_indices, val_k_indices) in enumerate(skf.split(x_train, y_train)):
        # Generate batches from indices
        x_train_k, x_val_k = x_data.iloc[train_k_indices], x_data.iloc[val_k_indices]
        y_train_k, y_val_k = y_data.iloc[train_k_indices], y_data.iloc[val_k_indices]

        # Clear model, and create it
        disc = Discriminator.create_model(feature_size)
        gen = Generator.create_model(feature_size)

        acc, disc_trained, gen_trained = __train_model(disc, gen, x_train_k, y_train_k, x_val_k, y_val_k, documents_data, queries_data)
        if acc > acc_best:
            disc_best = disc_trained
            gen_best = gen_trained

    return disc_best, gen_best, x_test, y_test


def __train_model(disc, gen, x_train, y_train, x_val, y_val, documents_data, queries_data):

    print('Start adversarial training')
    for epoch in range(params.DISC_TRAIN_EPOCHS):
        # Train Discriminator
        print('Training Discriminator ...')
        for d_epoch in range(params.DISC_TRAIN_GEN_EPOCHS):
            pos_neg_data = []
            pos_neg_size = 0
            if d_epoch % params.DISC_TRAIN_EPOCHS == 0:
                # Generator generate negative for Discriminator, then train Discriminator
                queries_ids = np.unique(x_train.query_id.astype(int))
                train_queries_data = queries_data[(queries_data.index.isin(queries_ids))]

                documents_ids = np.unique(x_train.doc_id.astype(int))
                train_documents_data = documents_data[(documents_data.index.isin(documents_ids))]

                pos_neg_data = __generate_negatives_for_discriminator(gen, x_train, y_train, train_documents_data, train_queries_data)
                pos_neg_size = len(pos_neg_data)

            i = 0
            while i < pos_neg_size:
                if i + params.DISC_BATCH_SIZE <= pos_neg_size + 1:
                    input_pos, input_neg = __get_batch_data(pos_neg_data, i, params.DISC_BATCH_SIZE)
                else:
                    input_pos, input_neg = __get_batch_data(pos_neg_data, i, pos_neg_size - i + 1)

                i += params.DISC_BATCH_SIZE

                # prepare pos and neg data
                pos_data = [[queries_data.loc[x[0]].query, documents_data.loc[x[1]].text] for x in input_pos]
                neg_data = [[queries_data.loc[x[0]].query, documents_data.loc[x[1]].text] for x in input_neg]

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
            gen.train_on_batch(x_train, y_train, sample_weight=None, class_weight=None)

    # Evaluate
    acc = 0.0

    return acc, disc, gen


def __generate_negatives_for_discriminator(gen, x_train, y_train, documents_data, queries_data):

    data = []

    print('negative sampling for discriminator using generator')
    for query_id in queries_data.index.values:
        # get all query specific ratings
        idx = (x_train.query_id == query_id)
        x_pos_list = x_train.loc[idx]
        y_pos_list = y_train.loc[idx]

        # get all other ratings
        docs_pos_ids = np.unique(x_pos_list.doc_id.astype(int))
        candidate_list = documents_data[(~documents_data.index.isin(docs_pos_ids))]

        # prob = gen.get_prob(candidate_list)
        # prob = prob[0]
        # prob = prob.reshape([-1])
        #
        # neg_list = np.random.choice(candidate_list.index, size=[len(x_pos_list)], p=prob)
        neg_list = np.random.choice(candidate_list.index, size=[len(x_pos_list)])

        for i in range(len(x_pos_list)):
            data.append((query_id, int(x_pos_list.iloc[i].doc_id), neg_list[i]))

    # shuffle
    random.shuffle(data)
    return data


def __get_batch_data(pos_neg_data, index, size):
    pos = []
    neg = []
    for i in range(index, index + size):
        line = pos_neg_data[i]
        pos.append([line[0], line[1]])
        neg.append([line[0], line[2]])
    return pos, neg

