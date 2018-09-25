import numpy as np
import random

import app.parameters as params
import app.data_preparation.mq2008.preprocessing_MQ2008 as mq


def generate_for_discriminator(generator, query_url_features, query_index_urls, query_pos_train, negative_file):
    data = []
    print('negative sampling for discriminator using generator')

    for query in query_pos_train:
        # get query all url (postive)
        pos_list = query_pos_train[query]
        # get query all url
        all_list = query_index_urls[query]

        candidate_list = all_list
        # get all url feature
        candidate_list_feature = [query_url_features[query][url] for url in candidate_list]
        candidate_list_feature = np.asarray(candidate_list_feature)
        """
        score = generator.get_score(candidate_list_feature[np.newaxis, :])
        score = score[0].reshape([-1])
        # softmax for all
        exp_rating = np.exp(score - np.max(score))
        prob = exp_rating / np.sum(exp_rating) 
        """
        prob = generator.get_prob(candidate_list_feature[np.newaxis, :])
        prob = prob[0]
        prob = prob.reshape([-1])
        # Generator generate some url (postive doc num)
        neg_list = np.random.choice(candidate_list, size=[len(pos_list)], p=prob)
        # list -> ( query id , pos url , neg url )
        for i in range(len(pos_list)):
            data.append((query, pos_list[i], neg_list[i]))
    # shuffle
    random.shuffle(data)
    with open(negative_file, 'w') as fout:
        # pos feature [tab] neg feature
        for (q, pos, neg) in data:
            fout.write(','.join([str(f) for f in query_url_features[q][pos]])
                       + '\t'
                       + ','.join([str(f) for f in query_url_features[q][neg]]) + '\n')
            fout.flush()


def train(generator, discriminator, query_url_features, query_urls, query_index_urls, query_pos_train, query_pos_train_file_size):
    for d_epoch in range(params.DISC_TRAIN_EPOCHS):

        if d_epoch % 30 == 0:
            generate_for_discriminator(generator, query_url_features, query_index_urls, query_pos_train, params.TRAIN_DATA_FILE)
            train_size = query_pos_train_file_size

        index = 1
        while True:
            if index > train_size:
                break
            if index + params.DISC_BATCH_SIZE <= train_size + 1:
                input_pos, input_neg = mq.get_batch_data(params.TRAIN_DATA_FILE, index, params.DISC_BATCH_SIZE)
            else:
                input_pos, input_neg = mq.get_batch_data(params.TRAIN_DATA_FILE, index, train_size - index + 1)
            index += params.DISC_BATCH_SIZE

            pred_data = []
            # prepare pos and neg data
            pred_data.extend(input_pos)
            pred_data.extend(input_neg)
            pred_data = np.asarray(pred_data)
            # prepara pos and neg label
            pred_data_label = [1.0] * len(input_pos)
            pred_data_label.extend([0.0] * len(input_neg))
            pred_data_label = np.asarray(pred_data_label)
            # train
            discriminator.train(pred_data, pred_data_label)
