import numpy as np
import random

from elasticsearch import Elasticsearch
from keras.layers import Embedding
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import parameters as params
from gan.adverserial_nn.generator import Generator
from gan.adverserial_nn.discriminator import Discriminator
from gan.adverserial_nn.generator_pretrain import GeneratorPretrain
from gan.layers import init_w2v_embeddings, init_fasttext_model_embeddings
import evaluation.metrics.precision_k as p_k
import evaluation.metrics.ndcg_k as ndcg_k


def get_x_data_splitted(query_ids):
    x_train, x_test = train_test_split(query_ids, test_size=0.10, random_state=42)
    return x_train, x_test


def train_model(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, weight_decay,
                learning_rate, temperature, dropout, experiment=None):
    train_x_indices, test_x_indices = get_x_data_splitted(x_train)

    # Generate batches from indices
    x_train_k, x_test_k = np.array(train_x_indices), np.array(test_x_indices)

    gen_pre, disc_pre = __pretrain_model(x_train_k, ratings_data, queries_data, documents_data, tokenizer_q,
                                         tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout,
                                         experiment)

    gen, disc, p_val, ndcg_val = __train_model(gen_pre, disc_pre, x_train_k, x_test_k, ratings_data, queries_data,
                                               documents_data,
                                               tokenizer_q, tokenizer_d, sess, weight_decay, learning_rate, temperature,
                                               dropout, experiment)

    return gen, disc, p_val, ndcg_val


def __get_embedding_layers(tokenizer_q, tokenizer_d) -> (Embedding, Embedding):
    if params.USE_FASTTEXT_MODEL:
        print('Load embeddings')
        embedding_model = init_fasttext_model_embeddings.load_model()
        print('Prepare embedding-layer for queries')
        embedding_layer_q = init_fasttext_model_embeddings.init_embedding_layer(tokenizer_q, embedding_model,
                                                                                params.MAX_SEQUENCE_LENGTH_QUERIES,
                                                                                params.MAX_NUM_WORDS_QUERIES)
        print('Prepare embedding-layer for documents')
        embedding_layer_d = init_fasttext_model_embeddings.init_embedding_layer(tokenizer_d, embedding_model,
                                                                                params.MAX_SEQUENCE_LENGTH_DOCS,
                                                                                params.MAX_NUM_WORDS_DOCS)
    else:
        print('Load embeddings')
        embedding_index = init_w2v_embeddings.build_index_mapping()
        print('Prepare embedding-layer for queries')
        embedding_layer_q = init_w2v_embeddings.init_embedding_layer(tokenizer_q, embedding_index,
                                                                     params.MAX_SEQUENCE_LENGTH_QUERIES,
                                                                     params.MAX_NUM_WORDS_QUERIES)
        print('Prepare embedding-layer for documents')
        embedding_layer_d = init_w2v_embeddings.init_embedding_layer(tokenizer_d, embedding_index,
                                                                     params.MAX_SEQUENCE_LENGTH_DOCS,
                                                                     params.MAX_NUM_WORDS_DOCS)
    return embedding_layer_q, embedding_layer_d


def __pretrain_model(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess, weight_decay,
                     learning_rate, temperature, dropout, experiment=None):
    train_ratings_data, train_queries_data, train_documents_data = __build_train_data(x_train, ratings_data,
                                                                                      queries_data, documents_data)

    # Clear models, and reinitialize them
    embedding_layer_q, embedding_layer_d = __get_embedding_layers(tokenizer_q, tokenizer_d)

    print('Build discriminator network')
    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    disc = Discriminator.create_model(samples_per_epoc, weight_decay, learning_rate, dropout, embedding_layer_q,
                                      embedding_layer_d, sess=sess)

    print('Build generator network')
    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    gen_pretrain = GeneratorPretrain.create_model(samples_per_epoc, weight_decay, learning_rate, temperature, dropout,
                                                  embedding_layer_q, embedding_layer_d, sess=sess)

    print('Start pre-model training')

    # Train Generator
    print('Training Generator ...')

    # Generator generate negative for Discriminator, then train Discriminator
    pos_neg_data = __generate_negatives_for_discriminator_pretrain(x_train, train_ratings_data, documents_data)

    input_pos, input_neg = __get_pos_neg_data(pos_neg_data)

    # prepare pos and neg data
    pos_data_queries = [queries_data[x[0]] for x in input_pos]
    pos_data_documents = [documents_data[x[1]] for x in input_pos]
    neg_data_queries = [queries_data[x[0]] for x in input_neg]
    neg_data_documents = [documents_data[x[1]] for x in input_neg]

    pos_data_queries = np.asarray(pos_data_queries)
    neg_data_queries = np.asarray(neg_data_queries)
    queries_gen = np.append(pos_data_queries, neg_data_queries)

    pos_data_documents = np.asarray(pos_data_documents)
    neg_data_documents = np.asarray(neg_data_documents)
    documents_gen = np.append(pos_data_documents, neg_data_documents)

    # prepare pos and neg label
    pos_data_label = [1.0] * len(pos_data_queries)
    pos_data_label = np.asarray(pos_data_label)
    pos_data_label = to_categorical(pos_data_label, 2)

    neg_data_label = [0.0] * len(neg_data_queries)
    neg_data_label = np.asarray(neg_data_label)
    neg_data_label = to_categorical(neg_data_label, 2)

    labels_gen = np.append(pos_data_label, neg_data_label)

    randomize = np.arange(len(queries_gen))
    np.random.shuffle(randomize)
    queries_gen = queries_gen[randomize]
    documents_gen = documents_gen[randomize]
    labels_gen = labels_gen[randomize]

    print("Pretrain Generator on batches of size: ", params.GEN_BATCH_SIZE)

    # train
    g_loss = gen_pretrain.train(queries_gen, documents_gen, labels_gen)
    print("g_loss:", g_loss)

    # Plot the progress
    g_acc = 100 * g_loss[1]
    g_loss_val = g_loss[0]

    print("[G loss: %f, acc.: %.2f%%]" % (g_loss_val, g_acc))
    experiment.log_metric("pretrain_gen_accuracy", g_acc)
    experiment.log_metric("pretrain_gen_loss", g_loss_val)

    gen_pretrain.save_model_to_weights(params.SAVED_MODEL_GEN_JSON, params.SAVED_MODEL_GEN_WEIGHTS)

    # Train Discriminator
    print('Training Discriminator ...')

    # Get similar negatives for Discriminator, then train Discriminator
    pos_neg_data = __generate_negatives_for_discriminator_pretrain(x_train, train_ratings_data, documents_data)

    input_pos, input_neg = __get_pos_neg_data(pos_neg_data)

    # prepare pos and neg data
    pos_data_queries = [queries_data[x[0]] for x in input_pos]
    pos_data_documents = [documents_data[x[1]] for x in input_pos]
    neg_data_queries = [queries_data[x[0]] for x in input_neg]
    neg_data_documents = [documents_data[x[1]] for x in input_neg]

    pos_data_queries = np.asarray(pos_data_queries)
    neg_data_queries = np.asarray(neg_data_queries)
    queries_disc = np.append(pos_data_queries, neg_data_queries)

    pos_data_documents = np.asarray(pos_data_documents)
    neg_data_documents = np.asarray(neg_data_documents)
    documents_disc = np.append(pos_data_documents, neg_data_documents)

    # prepare pos and neg label
    pos_data_label = [1.0] * len(pos_data_queries)
    pos_data_label = np.asarray(pos_data_label)
    neg_data_label = [0.0] * len(neg_data_queries)
    neg_data_label = np.asarray(neg_data_label)

    labels_disc = np.append(pos_data_label, neg_data_label)

    randomize = np.arange(len(queries_disc))
    np.random.shuffle(randomize)
    queries_disc = queries_disc[randomize]
    documents_disc = documents_disc[randomize]
    labels_disc = labels_disc[randomize]

    print("Pretrain Discriminator on batches of size: ", params.DISC_BATCH_SIZE)

    # train
    d_loss = disc.train(queries_disc, documents_disc, labels_disc)
    print("d_loss:", d_loss)

    # Plot the progress
    d_acc = 100 * d_loss[1]
    d_loss_val = d_loss[0]

    print("[D loss: %f, acc.: %.2f%%]" % (d_loss_val, d_acc))
    experiment.log_metric("pretrain_disc_accuracy", d_acc)
    experiment.log_metric("pretrain_disc_loss", d_loss_val)

    return gen_pretrain, disc


def __train_model(gen_pre, disc_pre, x_train, x_val, ratings_data, queries_data, documents_data, tokenizer_q,
                  tokenizer_d, sess, weight_decay, learning_rate, temperature, dropout, experiment=None):
    train_ratings_data, train_queries_data, train_documents_data = __build_train_data(x_train, ratings_data,
                                                                                      queries_data, documents_data)

    disc = disc_pre

    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    embedding_layer_q, embedding_layer_d = __get_embedding_layers(tokenizer_q, tokenizer_d)

    gen = Generator.create_model(samples_per_epoc, weight_decay, learning_rate, temperature, dropout,
                                 embedding_layer_q, embedding_layer_d, sess=sess)

    gen = gen.load_weights_for_model(params.SAVED_MODEL_GEN_WEIGHTS)

    # Initialize data for eval
    p_best_val = 0.0
    ndcg_best_val = 0.0

    best_disc = disc_pre
    best_gen = gen_pre

    print('Start adversarial training')
    for epoch in range(params.DISC_TRAIN_EPOCHS):

        # Train Generator
        # -------------------------------------------------------------------------------------------------------------#
        print('Training Generator ...')

        pos_neg_data = __generate_negatives_for_generator(best_gen, x_train, train_ratings_data, queries_data,
                                                              documents_data)

        queries_all = []
        documents_all = []
        doc_ids_all = []

        for query_id in x_train:

            # get all query specific ratings
            x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)
            x_pos_set = set(x_pos_list)

            doc_ids, data_queries, data_documents = __get_rand_batch_from_candidates_for_generator(query_id,
                                                                                                   queries_data,
                                                                                                   documents_data,
                                                                                                   candidate_list,
                                                                                                   x_pos_list)

            queries_all.extend(data_queries)
            documents_all.extend(data_documents)
            doc_ids_all.extend(doc_ids)



        # Importance Sampling
        prob = gen.get_prob(data_queries, data_documents)
            print("prob of gen: " + str(prob))

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

            print("reward / imp_sampling:")
            print(choose_reward)
            print(choose_is)

            choose_queries_all.append(choose_queries)
            choose_documents_all.append(choose_documents)
            choose_reward_all.append(choose_reward)
            choose_is_all.append(choose_is)

        choose_queries_all = np.asarray(choose_queries_all)
        choose_documents_all = np.asarray(choose_documents_all)
        choose_reward_all = np.asarray(choose_reward_all)
        choose_is_all = np.asarray(choose_is_all)

        # train
        g_loss = gen.train(choose_queries_all, choose_documents_all, choose_reward_all.reshape([-1]), choose_is_all)
        print("g_loss:", g_loss)

        # Plot the progress
        g_acc = 100 * g_loss[1]
        g_loss_val = g_loss[0]

        print("[G loss: %f, acc.: %.2f%%]" % (g_loss_val, g_acc))
        experiment.log_metric("gen_accuracy", g_acc)
        experiment.log_metric("gen_loss", g_loss_val)

        # Train Discriminator
        # -------------------------------------------------------------------------------------------------------------#
        print('Training Discriminator ...')

        # Generator generate negative for Discriminator, then train Discriminator
        pos_neg_data = __generate_negatives_for_discriminator(best_gen, x_train, train_ratings_data, queries_data, documents_data)
        input_pos, input_neg = __get_pos_neg_data(pos_neg_data)

        # prepare pos and neg data
        pos_data_queries = [queries_data[x[0]] for x in input_pos]
        pos_data_documents = [documents_data[x[1]] for x in input_pos]
        neg_data_queries = [queries_data[x[0]] for x in input_neg]
        neg_data_documents = [documents_data[x[1]] for x in input_neg]

        pos_data_queries = np.asarray(pos_data_queries)
        neg_data_queries = np.asarray(neg_data_queries)
        queries_disc = np.append(pos_data_queries, neg_data_queries)

        pos_data_documents = np.asarray(pos_data_documents)
        neg_data_documents = np.asarray(neg_data_documents)
        documents_disc = np.append(pos_data_documents, neg_data_documents)

        # prepare pos and neg label
        pos_data_label = [1.0] * len(pos_data_queries)
        pos_data_label = np.asarray(pos_data_label)
        neg_data_label = [0.0] * len(neg_data_queries)
        neg_data_label = np.asarray(neg_data_label)

        labels_disc = np.append(pos_data_label, neg_data_label)

        randomize = np.arange(len(queries_disc))
        np.random.shuffle(randomize)
        queries_disc = queries_disc[randomize]
        documents_disc = documents_disc[randomize]
        labels_disc = labels_disc[randomize]

        print("Train Discriminator on batches of size: ", params.DISC_BATCH_SIZE)

        # train
        d_loss = disc.train(queries_disc, documents_disc, labels_disc)
        print("d_loss:", d_loss)

        # Plot the progress
        d_acc = 100 * d_loss[1]
        d_loss_val = d_loss[0]

        print("[D loss: %f, acc.: %.2f%%]" % (d_loss_val, d_acc))
        experiment.log_metric("disc_accuracy", d_acc)
        experiment.log_metric("disc_loss", d_loss_val)

        # Evaluate
        # -------------------------------------------------------------------------------------------------------------#
        print('Evaluate models')

        p_step = p_k.measure_precision_at_k(disc, x_val, ratings_data, queries_data, documents_data, params.EVAL_K,
                                            sess)

        ndcg_step = ndcg_k.measure_ndcg_at_k(disc, x_val, ratings_data, queries_data, documents_data, params.EVAL_K,
                                             sess)

        print("Epoch", epoch, "measure:", "disc p@5 =", p_step, "disc ndcg@5 =", ndcg_step)
        experiment.log_metric("disc_p5", p_step, epoch)
        experiment.log_metric("disc_ndcg5", ndcg_step, epoch)

        best_disc, best_gen, p_best_val, ndcg_best_val = __get_best_eval_result(disc, best_disc, gen, best_gen, p_step,
                                                                                p_best_val, ndcg_step, ndcg_best_val)

    print("Best:", "disc p@5 =", p_best_val, "disc ndcg@5 =", ndcg_best_val)

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


def __get_pos_neg_data(pos_neg_data):
    pos = []
    neg = []
    for i in len(pos_neg_data):
        line = pos_neg_data[i]
        pos.append([line[0], line[1]])
        neg.append([line[0], line[2]])
    return pos, neg


def __generate_negatives_for_discriminator_pretrain(x_train, ratings_data, documents_data):
    data = []

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        doc_ids = __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, 3 * len(x_pos_list))

        neg_list = np.random.choice(doc_ids, size=[len(x_pos_list)])

        for i in range(len(x_pos_list)):
            data.append([query_id, x_pos_list[i], neg_list[i]])

    # shuffle
    random.shuffle(data)
    return data


def __generate_negatives_for_discriminator(gen, x_train, ratings_data, queries_data, documents_data):
    data = []

    pos_data = {}
    neg_data = {}

    neg_data_q_ids = []
    neg_data_queries = []
    neg_data_documents = []

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        candidate_ids = __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, 3 * len(x_pos_list))

        pos_data[query_id] = x_pos_list
        neg_data[query_id] = candidate_ids

        # prepare neg data
        cand_queries = [queries_data[query_id]] * len(candidate_ids)
        cand_documents = [documents_data[x] for x in candidate_ids]
        cand_q_ids = [query_id] * len(candidate_ids)

        neg_data_queries = np.append(neg_data_queries, cand_queries)
        neg_data_documents = np.append(neg_data_documents, cand_documents)
        neg_data_q_ids = np.append(neg_data_q_ids, cand_q_ids)

    # Importance Sampling
    probs = gen.get_prob(neg_data_queries, neg_data_documents)
    print("__get_rand_batch_from_candidates_for_negatives: prob = ", probs)

    probs = np.asarray(probs) / np.asarray(probs).sum(axis=0, keepdims=1)

    neg_data_cands = {}
    for i, prob in enumerate(probs):
        query_id = neg_data_q_ids[i]
        if query_id in neg_data_cands.keys():
            neg_data_cands[query_id] = np.append(neg_data_cands[query_id], [prob])
        else:
            neg_data_cands[query_id] = [prob]

    for query_id, probs in neg_data_cands.items():
        neg_list = np.random.choice(neg_data[query_id], size=[len(pos_data[query_id])], p=probs)
        neg_data[query_id] = neg_list

    for query_id, pos_values in pos_data.items():
        for i, pos_elem in enumerate(pos_values):
            neg_elem = neg_data[query_id][i]
            data.append([query_id, pos_elem, neg_elem])

    # shuffle
    random.shuffle(data)
    return data

def __generate_negatives_for_generator(gen, x_train, ratings_data, queries_data, documents_data):
    data = []

    pos_neg_data = {}

    pos_neg_data_q_ids = []
    pos_neg_data_queries = []
    pos_neg_data_documents = []

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        candidate_ids = __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, 2 * len(x_pos_list))

        pos_neg_data[query_id] = np.append(x_pos_list, candidate_ids)

        # prepare neg data
        cand_queries = [queries_data[query_id]] * len(pos_neg_data[query_id])
        cand_documents = [documents_data[x] for x in pos_neg_data[query_id]]
        cand_q_ids = [query_id] * len(pos_neg_data[query_id])

        pos_neg_data_queries = np.append(pos_neg_data_queries, cand_queries)
        pos_neg_data_documents = np.append(pos_neg_data_documents, cand_documents)
        pos_neg_data_q_ids = np.append(pos_neg_data_q_ids, cand_q_ids)

    # Importance Sampling
    prob = gen.get_prob(data_queries, data_documents)
    print("prob of gen: " + str(prob))

    print('start important sampling')
    # important sampling, change doc prob
    prob_is = prob * (1.0 - params.GEN_LAMBDA)

    for i in range(len(doc_ids)):
        if doc_ids[i] in x_pos_set:
            prob_is[i] += (params.GEN_LAMBDA / (1.0 * len(x_pos_list)))

    # G generate some url (2 * postive doc num)
    prob_is_rand = np.asarray(prob_is) / np.asarray(prob_is).sum(axis=0, keepdims=1)
    choose_index = np.random.choice(np.arange(len(doc_ids)), [2 * len(x_pos_list)], p=prob_is_rand)

    # Importance Sampling
    probs = gen.get_prob(pos_neg_data_queries, pos_neg_data_documents)
    print("__get_rand_batch_from_candidates_for_negatives: prob = ", probs)

    probs = np.asarray(probs) / np.asarray(probs).sum(axis=0, keepdims=1)

    neg_data_cands = {}
    for i, prob in enumerate(probs):
        query_id = neg_data_q_ids[i]
        if query_id in neg_data_cands.keys():
            neg_data_cands[query_id] = np.append(neg_data_cands[query_id], [prob])
        else:
            neg_data_cands[query_id] = [prob]

    for query_id, probs in neg_data_cands.items():
        neg_list = np.random.choice(neg_data[query_id], size=[len(pos_data[query_id])], p=probs)
        neg_data[query_id] = neg_list

    for query_id, pos_values in pos_data.items():
        for i, pos_elem in enumerate(pos_values):
            neg_elem = neg_data[query_id][i]
            data.append([query_id, pos_elem, neg_elem])

    # shuffle
    random.shuffle(data)
    return data

def __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, size):
    # create ES client, create index
    es = Elasticsearch(hosts=[params.ES_HOST])
    query_text = es.get(index="queries", doc_type="query", id=str(query_id))["_source"]["text"]
    candidates = es.search(index="documents", body={"query": {"match": {"text": query_text}}}, size=size)
    candidates = [doc['_id'] for doc in candidates['hits']['hits']]

    # prepare pos and neg data
    doc_ids = np.array(candidates)

    if len(candidates) < size:
        candidates_addition = np.random.choice(candidate_list, size=[size - len(candidates)])
        doc_ids = np.append(doc_ids, candidates_addition)

    return doc_ids


# def __get_rand_batch_from_candidates_for_generator(query_id, queries_data, documents_data, candidate_list,
#                                                    x_pos_list):
#     # create ES client, create index
#     es = Elasticsearch(hosts=[params.ES_HOST])
#     query_text = es.get(index="queries", doc_type="query", id=str(query_id))["_source"]["text"]
#     size = 2 * len(x_pos_list)
#     candidates = es.search(index="documents", body={"query": {"match": {"text": query_text}}}, size=size)
#     candidates = [doc['_id'] for doc in candidates['hits']['hits']]
#
#     # prepare pos and neg data
#     doc_ids = np.array(candidates)
#
#     if len(candidates) < size:
#         candidates_addition = np.random.choice(candidate_list, size=[size - len(candidates)])
#         doc_ids = np.append(doc_ids, candidates_addition)
#
#     # prepare pos and neg data
#     data_queries = [queries_data[query_id]] * (3 * len(x_pos_list))
#
#     data_documents_cand = [documents_data[x] for x in doc_ids]
#     data_documents_pos = [documents_data[x] for x in x_pos_list]
#
#     data_documents = []
#     data_documents.extend(data_documents_cand)
#     data_documents.extend(data_documents_pos)
#
#     doc_ids_collected = []
#     doc_ids_collected.extend(doc_ids)
#     doc_ids_collected.extend(x_pos_list)
#
#     return doc_ids_collected, data_queries, data_documents


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
