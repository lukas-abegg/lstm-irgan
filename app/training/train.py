import numpy as np
import random

from keras.layers import Embedding

from sklearn.model_selection import train_test_split

import parameters as params
from gan.adverserial_nn.generator import Generator
from gan.adverserial_nn.discriminator import Discriminator
from gan.layers import init_w2v_embeddings, init_fasttext_model_embeddings
import evaluation.metrics.precision_k as p_k
import evaluation.metrics.ndcg_k as ndcg_k


def get_x_data_splitted(query_ids):
    x_train, x_test = train_test_split(query_ids, test_size=0.10, random_state=42)
    return x_train, x_test


def train_model(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout, experiment=None):

    train_x_indices, test_x_indices = get_x_data_splitted(x_train)

    # Generate batches from indices
    x_train_k, x_test_k = np.array(train_x_indices), np.array(test_x_indices)

    gen_pre, disc_pre = __pretrain_model(x_train_k, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout, experiment)

    gen, disc, p_val, ndcg_val = __train_model(gen_pre, disc_pre, x_train_k, x_test_k, ratings_data, queries_data, documents_data,
                                                   tokenizer_q, tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout, experiment)

    return gen, disc, p_val, ndcg_val


def __get_embedding_layers(tokenizer_q, tokenizer_d) -> (Embedding, Embedding):
    if params.USE_FASTTEXT_MODEL:
        print('Load embeddings')
        embedding_model = init_fasttext_model_embeddings.load_model()
        print('Prepare embedding-layer for queries')
        embedding_layer_q = init_fasttext_model_embeddings.init_embedding_layer(tokenizer_q, embedding_model,
                                                                     params.MAX_SEQUENCE_LENGTH)
        print('Prepare embedding-layer for documents')
        embedding_layer_d = init_fasttext_model_embeddings.init_embedding_layer(tokenizer_d, embedding_model,
                                                                     params.MAX_SEQUENCE_LENGTH)
    else:
        print('Load embeddings')
        embedding_index = init_w2v_embeddings.build_index_mapping()
        print('Prepare embedding-layer for queries')
        embedding_layer_q = init_w2v_embeddings.init_embedding_layer(tokenizer_q, embedding_index,
                                                                     params.MAX_SEQUENCE_LENGTH)
        print('Prepare embedding-layer for documents')
        embedding_layer_d = init_w2v_embeddings.init_embedding_layer(tokenizer_d, embedding_index,
                                                                     params.MAX_SEQUENCE_LENGTH)
    return embedding_layer_q, embedding_layer_d


def __pretrain_model(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout, experiment=None):

    train_ratings_data, train_queries_data, train_documents_data = __build_train_data(x_train, ratings_data, queries_data, documents_data)

    # Clear models, and reinitialize them
    embedding_layer_q, embedding_layer_d = __get_embedding_layers(tokenizer_q, tokenizer_d)

    print('Build discriminator network')
    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    disc = Discriminator.create_model(samples_per_epoc, weight_decay, learning_rate, dropout, embedding_layer_q, embedding_layer_d, sess=sess)

    print('Build generator network')
    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    gen = Generator.create_model(samples_per_epoc, weight_decay, learning_rate, temperature, dropout, embedding_layer_q, embedding_layer_d, sess=sess)

    print('Start pre-model training')
    # Train Discriminator
    print('Training Discriminator ...')
    for d_epoch in range(params.DISC_TRAIN_GEN_EPOCHS):
        print('now_ D_epoch : ', str(d_epoch))

        pos_neg_data = []
        pos_neg_size = 0
        if d_epoch % params.DISC_TRAIN_EPOCHS == 0:
            # Generator generate negative for Discriminator, then train Discriminator
            pos_neg_data = __generate_negatives_for_discriminator_pretrain(x_train, train_ratings_data, queries_data, documents_data)
            pos_neg_size = len(pos_neg_data)

        print('train on batches of size: ', params.DISC_BATCH_SIZE)
        i = 1
        while i <= pos_neg_size:
            batch_index = i - 1
            if i + params.DISC_BATCH_SIZE <= pos_neg_size:
                input_pos, input_neg = __get_batch_data(pos_neg_data, batch_index, params.DISC_BATCH_SIZE)
            else:
                input_pos, input_neg = __get_batch_data(pos_neg_data, batch_index, pos_neg_size - batch_index)

            i += params.DISC_BATCH_SIZE

            # prepare pos and neg data
            pos_data_queries = [queries_data[x[0]] for x in input_pos]
            pos_data_documents = [documents_data[x[1]] for x in input_pos]
            neg_data_queries = [queries_data[x[0]] for x in input_neg]
            neg_data_documents = [documents_data[x[1]] for x in input_neg]

            pos_data_queries = np.asarray(pos_data_queries)
            neg_data_queries = np.asarray(neg_data_queries)

            pos_data_documents = np.asarray(pos_data_documents)
            neg_data_documents = np.asarray(neg_data_documents)

            # prepare pos and neg label
            pos_data_label = [1.0] * len(pos_data_queries)
            pos_data_label = np.asarray(pos_data_label)
            neg_data_label = [0.0] * len(neg_data_queries)
            neg_data_label = np.asarray(neg_data_label)

            print("Discriminator epoch: ", str(d_epoch), "with batch: ", str(batch_index), " to ", str(i-1), " of ", str(pos_neg_size))
            # train
            d_loss_real = disc.train(pos_data_queries, pos_data_documents, pos_data_label)
            d_loss_fake = disc.train(neg_data_queries, neg_data_documents, neg_data_label)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Plot the progress
            d_acc = 100 * d_loss[1]
            d_loss_val = d_loss[0]

            print("%s [D loss: %f, acc.: %.2f%%]" % (str(batch_index)+"_"+str(i - 1), d_loss_val, d_acc))
            experiment.log_metric("pretrain_disc_accuracy", d_acc, i-1)
            experiment.log_metric("pretrain_disc_loss", d_loss_val, i-1)

    # Train Generator
    print('Training Generator ...')
    for g_epoch in range(params.GEN_TRAIN_EPOCHS):
        print('now_ G_epoch : ', str(g_epoch))

        x = 0
        len_queries = len(x_train)

        for query_id in x_train:

            # get all query specific ratings
            x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)
            x_pos_set = set(x_pos_list)

            prob, doc_ids, data_queries, data_documents = __get_rand_batch_from_candidates_for_generator(gen, query_id,
                                                                                                queries_data,
                                                                                                documents_data,
                                                                                                candidate_list,
                                                                                                x_pos_list)

            # important sampling, change doc prob
            prob_is = prob * (1.0 - params.GEN_LAMBDA)

            print("prob_is of gen: " + str(prob_is))

            for i in range(len(doc_ids)):
                if doc_ids[i] in x_pos_set:
                    prob_is[i] += (params.GEN_LAMBDA / (1.0 * len(x_pos_list)))

            print("prob_is of gen after lambda: " + str(prob_is))

            # G generate some url (2 * postive doc num)
            prob_is_rand = np.asarray(prob_is) / np.asarray(prob_is).sum(axis=0, keepdims=1)
            choose_index = np.random.choice(np.arange(len(doc_ids)), [2 * len(x_pos_list)], p=prob_is_rand)

            # choose data
            choose_queries = np.array(data_queries)[choose_index]
            choose_documents = np.array(data_documents)[choose_index]

            # prob / important sampling prob (loss => prob * reward * prob / important sampling prob)
            choose_is = np.array(prob)[choose_index] / np.array(prob_is)[choose_index]

            choose_queries = np.asarray(choose_queries)
            choose_documents = np.asarray(choose_documents)

            choose_is = np.asarray(choose_is)

            # get reward((prob  - 0.5) * 2 )
            choose_reward = disc.get_reward(choose_queries, choose_documents)

            x += 1
            print("Generator epoch: ", str(g_epoch), " with query: ", str(x), " of ", str(len_queries))
            # train
            g_loss = gen.train(choose_queries, choose_documents, choose_reward.reshape([-1]), choose_is)

            # Plot the progress
            g_acc = 100 * g_loss[1]
            g_loss_val = g_loss[0]

            print("%s [G loss: %f, acc.: %.2f%%]" % (str(x), g_loss_val, g_acc))
            experiment.log_metric("pretrain_gen_accuracy", g_acc, x)
            experiment.log_metric("pretrain_gen_loss", g_loss_val, x)

    return gen, disc


def __train_model(gen_pre, disc_pre, x_train, x_val, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout, experiment=None):
    train_ratings_data, train_queries_data, train_documents_data = __build_train_data(x_train, ratings_data, queries_data, documents_data)

    disc = disc_pre
    gen = gen_pre

    # Initialize data for eval
    p_best_val = 0.0
    ndcg_best_val = 0.0

    best_disc = disc_pre
    best_gen = gen_pre

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
                pos_neg_data = __generate_negatives_for_discriminator(best_gen, x_train, train_ratings_data, queries_data, documents_data)
                pos_neg_size = len(pos_neg_data)

            print('train on batches of size: ', params.DISC_BATCH_SIZE)
            i = 1
            while i <= pos_neg_size:
                batch_index = i - 1
                if i + params.DISC_BATCH_SIZE <= pos_neg_size:
                    input_pos, input_neg = __get_batch_data(pos_neg_data, batch_index, params.DISC_BATCH_SIZE)
                else:
                    input_pos, input_neg = __get_batch_data(pos_neg_data, batch_index, pos_neg_size - batch_index)

                i += params.DISC_BATCH_SIZE

                # prepare pos and neg data
                pos_data_queries = [queries_data[x[0]] for x in input_pos]
                pos_data_documents = [documents_data[x[1]] for x in input_pos]
                neg_data_queries = [queries_data[x[0]] for x in input_neg]
                neg_data_documents = [documents_data[x[1]] for x in input_neg]

                pos_data_queries = np.asarray(pos_data_queries)
                neg_data_queries = np.asarray(neg_data_queries)

                pos_data_documents = np.asarray(pos_data_documents)
                neg_data_documents = np.asarray(neg_data_documents)

                # prepare pos and neg label
                pos_data_label = [1.0] * len(pos_data_queries)
                pos_data_label = np.asarray(pos_data_label)
                neg_data_label = [0.0] * len(neg_data_queries)
                neg_data_label = np.asarray(neg_data_label)

                print("Discriminator epoch: ", str(d_epoch), "with batch: ", str(batch_index), " to ", str(i - 1),
                      " of ", str(pos_neg_size))
                # train
                d_loss_real = disc.train(pos_data_queries, pos_data_documents, pos_data_label)
                d_loss_fake = disc.train(neg_data_queries, neg_data_documents, neg_data_label)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Plot the progress
                d_acc = 100 * d_loss[1]
                d_loss_val = d_loss[0]

                print("%s [D loss: %f, acc.: %.2f%%]" % (str(batch_index)+"_"+str(i - 1), d_loss_val, d_acc))
                experiment.log_metric("disc_accuracy", d_acc, i - 1)
                experiment.log_metric("disc_loss", d_loss_val, i - 1)

        # Train Generator
        print('Training Generator ...')
        for g_epoch in range(params.GEN_TRAIN_EPOCHS):
            print('now_ G_epoch : ', str(g_epoch))

            x = 0
            len_queries = len(x_train)

            for query_id in x_train:

                # get all query specific ratings
                x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)
                x_pos_set = set(x_pos_list)

                prob, doc_ids, data_queries, data_documents = __get_rand_batch_from_candidates_for_generator(gen, query_id,
                                                                                                    queries_data,
                                                                                                    documents_data,
                                                                                                    candidate_list,
                                                                                                    x_pos_list)

                print('start important sampling')
                # important sampling, change doc prob
                prob_is = prob * (1.0 - params.GEN_LAMBDA)

                for i in range(len(doc_ids)):
                    if doc_ids[i] in x_pos_set:
                        prob_is[i] += (params.GEN_LAMBDA / (1.0 * len(x_pos_list)))

                # G generate some url (2 * postive doc num)
                prob_is_rand = np.asarray(prob_is) / np.asarray(prob_is).sum(axis=0, keepdims=1)
                choose_index = np.random.choice(np.arange(len(doc_ids)), [2 * len(x_pos_list)], p=prob_is_rand)

                # choose data
                choose_queries = np.array(data_queries)[choose_index]
                choose_documents = np.array(data_documents)[choose_index]

                # prob / important sampling prob (loss => prob * reward * prob / important sampling prob)
                choose_is = np.array(prob)[choose_index] / np.array(prob_is)[choose_index]

                choose_queries = np.asarray(choose_queries)
                choose_documents = np.asarray(choose_documents)

                choose_is = np.asarray(choose_is)

                # get reward((prob  - 0.5) * 2 )
                choose_reward = disc.get_reward(choose_queries, choose_documents)

                x += 1
                print("Generator epoch: ", str(g_epoch), " with query: ", str(x), " of ", str(len_queries))
                # train
                g_loss = gen.train(choose_queries, choose_documents, choose_reward.reshape([-1]), choose_is)

                # Plot the progress
                g_acc = 100 * g_loss[1]
                g_loss_val = g_loss[0]

                print("%s [G loss: %f, acc.: %.2f%%]" % (str(x), g_loss_val, g_acc))
                experiment.log_metric("gen_accuracy", g_acc, x)
                experiment.log_metric("gen_loss", g_loss_val, x)

            best_disc = disc
            best_gen = gen

            print('Evaluate models')
            # Evaluate
            p_step = p_k.measure_precision_at_k(gen, x_val, ratings_data, queries_data, documents_data, params.EVAL_K, sess)
            ndcg_step = ndcg_k.measure_ndcg_at_k(gen, x_val, ratings_data, queries_data, documents_data, params.EVAL_K, sess)

            print("Epoch", g_epoch, "measure:", "gen p@5 =", p_step, "gen ndcg@5 =", ndcg_step)
            experiment.log_metric("gen_p5", p_step, g_epoch)
            experiment.log_metric("gen_ndcg5", ndcg_step, g_epoch)

            best_disc, best_gen, p_best_val, ndcg_best_val = __get_best_eval_result(disc, best_disc, gen, best_gen, p_step,
                                                                                 p_best_val, ndcg_step, ndcg_best_val)

    print("Best:", "gen p@5 =", p_best_val, "gen ndcg@5 =", ndcg_best_val)

    return best_gen, best_disc, p_best_val, ndcg_best_val


def __build_train_data(x_train, ratings_data, queries_data, documents_data):
    train_queries_data = {}
    train_documents_data = {}
    train_ratings_data = {}

    for query_id in x_train:
        train_ratings_data[query_id] = ratings_data[query_id]
        train_queries_data[query_id] = queries_data[query_id]
        for key in ratings_data.keys():
            if key in documents_data.keys():
                train_documents_data[key] = documents_data[key]

    return train_ratings_data, train_queries_data, train_documents_data


def __generate_negatives_for_discriminator_pretrain(x_train, ratings_data, queries_data, documents_data):
    data = []

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        doc_ids, data_queries, data_documents = __get_rand_batch_from_candidates_for_negatives_pretrain(query_id, queries_data, documents_data, candidate_list, x_pos_list)

        neg_list = np.random.choice(doc_ids, size=[len(x_pos_list)])

        for i in range(len(x_pos_list)):
            data.append([query_id, x_pos_list[i], neg_list[i]])

    # shuffle
    random.shuffle(data)
    return data


def __generate_negatives_for_discriminator(gen, x_train, ratings_data, queries_data, documents_data):
    data = []

    x = 0
    len_queries = len(x_train)

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        prob, doc_ids, data_queries, data_documents = __get_rand_batch_from_candidates_for_negatives(gen, query_id, queries_data, documents_data, candidate_list, x_pos_list)

        prob = np.asarray(prob) / np.asarray(prob).sum(axis=0, keepdims=1)
        neg_list = np.random.choice(doc_ids, size=[len(x_pos_list)], p=prob)

        for i in range(len(x_pos_list)):
            data.append([query_id, x_pos_list[i], neg_list[i]])

        x += 1
        print("query: ", str(x), " of ", str(len_queries))

    # shuffle
    random.shuffle(data)
    return data


def __get_rand_batch_from_candidates_for_negatives_pretrain(query_id, queries_data, documents_data, candidate_list, x_pos_list):
    rand_batch = np.random.choice(np.arange(len(candidate_list)), [3 * len(x_pos_list)])

    # prepare pos and neg data
    data_queries = [queries_data[query_id]] * len(rand_batch)
    doc_ids = np.array(candidate_list)[rand_batch]
    data_documents = [documents_data[x] for x in doc_ids]

    return doc_ids, data_queries, data_documents


def __get_rand_batch_from_candidates_for_negatives(gen, query_id, queries_data, documents_data, candidate_list, x_pos_list):
    rand_batch = np.random.choice(np.arange(len(candidate_list)), [3 * len(x_pos_list)])

    # prepare pos and neg data
    data_queries = [queries_data[query_id]] * len(rand_batch)
    doc_ids = np.array(candidate_list)[rand_batch]
    data_documents = [documents_data[x] for x in doc_ids]

    # Importance Sampling
    prob = gen.get_prob(data_queries, data_documents)
    prob = prob.reshape([-1])

    return prob, doc_ids, data_queries, data_documents


def __get_rand_batch_from_candidates_for_generator(gen, query_id, queries_data, documents_data, candidate_list, x_pos_list):
    rand_batch = np.random.choice(np.arange(len(candidate_list)), [2 * len(x_pos_list)])

    # prepare pos and neg data
    data_queries = [queries_data[query_id]] * (3 * len(x_pos_list))
    doc_ids = np.array(candidate_list)[rand_batch]
    data_documents_cand = [documents_data[x] for x in doc_ids]
    data_documents_pos = [documents_data[x] for x in x_pos_list]

    data_documents = []
    data_documents.extend(data_documents_cand)
    data_documents.extend(data_documents_pos)

    # Importance Sampling
    prob = gen.get_prob(data_queries, data_documents)
    prob = prob.reshape([-1])
    print("prob of gen: "+str(prob))

    doc_ids_collected = []
    doc_ids_collected.extend(doc_ids)
    doc_ids_collected.extend(x_pos_list)

    return prob, doc_ids_collected, data_queries, data_documents


def __get_query_specific_data(query_id, ratings_data, documents_data):
    # get all query specific ratings
    x_pos_list = list(ratings_data[query_id].keys())[:params.POS_TRAINING_DATA_PER_QUERY]

    # get all other ratings
    docs_pos_ids = np.unique(x_pos_list)
    candidate_list = []
    for doc_id in documents_data.keys():
        if doc_id not in docs_pos_ids:
            candidate_list.append(doc_id)

    return x_pos_list, candidate_list


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
