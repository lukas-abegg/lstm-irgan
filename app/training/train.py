import numpy as np

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
                learning_rate_d, learning_rate_g, temperature, dropout, experiment=None):
    train_x_indices, test_x_indices = get_x_data_splitted(x_train)

    # Generate batches from indices
    x_train_k, x_test_k = np.array(train_x_indices), np.array(test_x_indices)

    gen_pre, disc_pre = __pretrain_model(x_train_k, ratings_data, queries_data, documents_data, tokenizer_q,
                                         tokenizer_d, sess, weight_decay, learning_rate_d, learning_rate_d, temperature, dropout,
                                         experiment)

    gen, disc, p_val, ndcg_val = __train_model(gen_pre, disc_pre, x_train_k, x_test_k, ratings_data, queries_data,
                                               documents_data,
                                               tokenizer_q, tokenizer_d, sess, weight_decay, learning_rate_d, learning_rate_g, temperature,
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
        # embedding_layer_q = Embedding(input_dim=len(tokenizer_q.word_index) + 1,
        #                               output_dim=params.EMBEDDING_DIM,
        #                               weights=None,
        #                               input_length=params.MAX_SEQUENCE_LENGTH_QUERIES,
        #                               mask_zero=True,
        #                               trainable=False)

        print('Prepare embedding-layer for documents')
        embedding_layer_d = init_fasttext_model_embeddings.init_embedding_layer(tokenizer_d, embedding_model,
                                                                                params.MAX_SEQUENCE_LENGTH_DOCS,
                                                                                params.MAX_NUM_WORDS_DOCS)
        # embedding_layer_d = Embedding(input_dim=len(tokenizer_q.word_index) + 1,
        #                               output_dim=params.EMBEDDING_DIM,
        #                               weights=None,
        #                               input_length=params.MAX_SEQUENCE_LENGTH_DOCS,
        #                               mask_zero=True,
        #                               trainable=False)
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
                     learning_rate_d, learning_rate_g, temperature, dropout, experiment=None):
    train_ratings_data, train_queries_data = __build_train_data(x_train, ratings_data, queries_data)

    # Clear models, and reinitialize them
    embedding_layer_q, embedding_layer_d = __get_embedding_layers(tokenizer_q, tokenizer_d)

    print('Build discriminator network')
    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    disc = Discriminator.create_model(samples_per_epoc, weight_decay, learning_rate_d, dropout, embedding_layer_q,
                                      embedding_layer_d, sess=sess)

    print('Build generator network')
    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    gen_pretrain = GeneratorPretrain.create_model(samples_per_epoc, weight_decay, learning_rate_g, temperature, dropout,
                                                  embedding_layer_q, embedding_layer_d, sess=sess)

    print('Start pre-model training')

    # Train Generator
    print('Training Generator ...')

    # Generator generate negative for Discriminator, then train Discriminator
    input_pos_gen_pre, input_neg_gen_pre = __generate_negatives_for_discriminator_pretrain(x_train, train_ratings_data, documents_data)

    # prepare pos and neg data
    pos_data_queries_gen_pre = [queries_data[x[0]] for x in input_pos_gen_pre]
    pos_data_documents_gen_pre = [documents_data[x[1]] for x in input_pos_gen_pre]
    neg_data_queries_gen_pre = [queries_data[x[0]] for x in input_neg_gen_pre]
    neg_data_documents_gen_pre = [documents_data[x[1]] for x in input_neg_gen_pre]

    queries_gen_pre = pos_data_queries_gen_pre[:]
    queries_gen_pre.extend(neg_data_queries_gen_pre)
    queries_gen_pre = np.asarray(queries_gen_pre)

    documents_gen_pre = pos_data_documents_gen_pre[:]
    documents_gen_pre.extend(neg_data_documents_gen_pre)
    documents_gen_pre = np.asarray(documents_gen_pre)

    # prepare pos and neg label
    pos_data_label_gen_pre = [1.0] * len(pos_data_queries_gen_pre)
    neg_data_label_gen_pre = [0.0] * len(neg_data_queries_gen_pre)

    labels_gen_pre = pos_data_label_gen_pre[:]
    labels_gen_pre.extend(neg_data_label_gen_pre)
    labels_gen_pre = np.asarray(labels_gen_pre)
    labels_gen_pre = to_categorical(labels_gen_pre, 2)

    randomize_gen_pre = np.arange(len(queries_gen_pre))
    np.random.shuffle(randomize_gen_pre)
    queries_gen_pre = queries_gen_pre[randomize_gen_pre]
    documents_gen_pre = documents_gen_pre[randomize_gen_pre]
    labels_gen_pre = labels_gen_pre[randomize_gen_pre]

    print("Pretrain Generator on batches of size: ", params.GEN_BATCH_SIZE)

    # train
    history = gen_pretrain.train(queries_gen_pre, documents_gen_pre, labels_gen_pre)

    # Plot the progress
    for i, epoch in enumerate(history.epoch):
        print("Epoch %s [G loss: %f, acc.: %.2f%%]" % (epoch, history.history["loss"][i], history.history["acc"][i]))
        step = int(epoch)
        experiment.log_metric('loss_pretrain_gen', history.history["loss"][i], step=step)
        experiment.log_metric('accuracy_pretrain_gen', history.history["acc"][i], step=step)

    gen_pretrain.save_model_to_weights(params.SAVED_MODEL_GEN_JSON, params.SAVED_MODEL_GEN_WEIGHTS)

    # Train Discriminator
    print('Training Discriminator ...')

    # Get similar negatives for Discriminator, then train Discriminator
    input_pos_disc_pre, input_neg_disc_pre = __generate_negatives_for_discriminator_pretrain(x_train, train_ratings_data, documents_data)

    # prepare pos and neg data
    pos_data_queries_disc_pre = [queries_data[x[0]] for x in input_pos_disc_pre]
    pos_data_documents_disc_pre = [documents_data[x[1]] for x in input_pos_disc_pre]
    neg_data_queries_disc_pre = [queries_data[x[0]] for x in input_neg_disc_pre]
    neg_data_documents_disc_pre = [documents_data[x[1]] for x in input_neg_disc_pre]

    queries_disc_pre = pos_data_queries_disc_pre[:]
    queries_disc_pre.extend(neg_data_queries_disc_pre)
    queries_disc_pre = np.asarray(queries_disc_pre)

    documents_disc_pre = pos_data_documents_disc_pre[:]
    documents_disc_pre.extend(neg_data_documents_disc_pre)
    documents_disc_pre = np.asarray(documents_disc_pre)

    # prepare pos and neg label
    pos_data_label_disc_pre = [1.0] * len(pos_data_queries_disc_pre)
    neg_data_label_disc_pre = [0.0] * len(neg_data_queries_disc_pre)

    labels_disc_pre = pos_data_label_disc_pre[:]
    labels_disc_pre.extend(neg_data_label_disc_pre)
    labels_disc_pre = np.asarray(labels_disc_pre)

    randomize_disc_pre = np.arange(len(queries_disc_pre))
    np.random.shuffle(randomize_disc_pre)
    queries_disc_pre = queries_disc_pre[randomize_disc_pre]
    documents_disc_pre = documents_disc_pre[randomize_disc_pre]
    labels_disc_pre = labels_disc_pre[randomize_disc_pre]

    print("Pretrain Discriminator on batches of size: ", params.DISC_BATCH_SIZE)

    # train
    history = disc.train(queries_disc_pre, documents_disc_pre, labels_disc_pre)

    # Plot the progress
    for i, epoch in enumerate(history.epoch):
        print("Epoch %s [G loss: %f, acc.: %.2f%%]" % (epoch, history.history["loss"][i], history.history["acc"][i]))
        step = int(epoch)
        experiment.log_metric('loss_pretrain_disc', history.history["loss"][i], step=step)
        experiment.log_metric('accuracy_pretrain_disc', history.history["acc"][i], step=step)

    return gen_pretrain, disc


def __train_model(gen_pre, disc_pre, x_train, x_val, ratings_data, queries_data, documents_data, tokenizer_q,
                  tokenizer_d, sess, weight_decay, learning_rate_d, learning_rate_g, temperature, dropout, experiment=None):
    train_ratings_data, train_queries_data = __build_train_data(x_train, ratings_data, queries_data)

    disc = disc_pre

    samples_per_epoc = len(x_train) * params.POS_TRAINING_DATA_PER_QUERY * 2
    embedding_layer_q, embedding_layer_d = __get_embedding_layers(tokenizer_q, tokenizer_d)

    gen = Generator.create_model(samples_per_epoc, weight_decay, learning_rate_g, temperature, dropout,
                                 embedding_layer_q, embedding_layer_d, sess=sess)

    gen = gen.load_weights_for_model(params.SAVED_MODEL_GEN_WEIGHTS)

    # initialize data for eval
    p_best_val = 0.0
    ndcg_best_val = 0.0

    best_disc = disc_pre
    best_gen = gen_pre

    last_gen = gen_pre

    print('Start adversarial training')
    for epoch in range(params.TRAIN_EPOCHS):

        print('Start adversial training for epoch:', epoch)

        # Train Generator
        # -------------------------------------------------------------------------------------------------------------#
        print('Training Generator ...')

        pos_neg_data_gen, pos_neg_probs_is_gen = __generate_negatives_for_generator(last_gen, x_train, train_ratings_data, queries_data,
                                                              documents_data)

        # choose data
        choose_queries_gen = [queries_data[x[0]] for x in pos_neg_data_gen]
        choose_documents_gen = [documents_data[x[1]] for x in pos_neg_data_gen]

        choose_queries_gen = np.asarray(choose_queries_gen)
        choose_documents_gen = np.asarray(choose_documents_gen)

        choose_is_gen = np.asarray(pos_neg_probs_is_gen)

        # get reward((prob  - 0.5) * 2 )
        choose_reward_gen = disc.get_reward(choose_queries_gen, choose_documents_gen)

        print("reward / imp_sampling:")
        print(choose_reward_gen)
        print(choose_is_gen)

        print("Train Generator on batches of size: ", params.GEN_BATCH_SIZE)

        # train
        history = gen.train(choose_queries_gen, choose_documents_gen, choose_reward_gen.reshape([-1]), choose_is_gen)

        # Plot the progress
        for i, epoch_gen in enumerate(history.epoch):
            print("Epoch %s [G loss: %f, acc.: %.2f%%]" % (epoch_gen, history.history["loss"][i], history.history["acc"][i]))
            step = int(epoch * 10 + epoch_gen)
            experiment.log_metric('loss_train_gen', history.history["loss"][i], step=step)
            experiment.log_metric('accuracy_train_gen', history.history["acc"][i], step=step)

        # Train Discriminator
        # -------------------------------------------------------------------------------------------------------------#
        print('Training Discriminator ...')

        # Generator generate negative for Discriminator, then train Discriminator
        input_pos_disc, input_neg_disc = __generate_negatives_for_discriminator(last_gen, x_train, train_ratings_data, queries_data, documents_data)

        # prepare pos and neg data
        pos_data_queries_disc = [queries_data[x[0]] for x in input_pos_disc]
        pos_data_documents_disc = [documents_data[x[1]] for x in input_pos_disc]
        neg_data_queries_disc = [queries_data[x[0]] for x in input_neg_disc]
        neg_data_documents_disc = [documents_data[x[1]] for x in input_neg_disc]

        queries_disc = pos_data_queries_disc[:]
        queries_disc.extend(neg_data_queries_disc)
        queries_disc = np.asarray(queries_disc)

        documents_disc = pos_data_documents_disc[:]
        documents_disc.extend(neg_data_documents_disc)
        documents_disc = np.asarray(documents_disc)

        # prepare pos and neg label
        pos_data_label_disc = [1.0] * len(pos_data_queries_disc)
        neg_data_label_disc = [0.0] * len(neg_data_queries_disc)

        labels_disc = pos_data_label_disc[:]
        labels_disc.extend(neg_data_label_disc)
        labels_disc = np.asarray(labels_disc)

        randomize_disc = np.arange(len(queries_disc))
        np.random.shuffle(randomize_disc)
        queries_disc = queries_disc[randomize_disc]
        documents_disc = documents_disc[randomize_disc]
        labels_disc = labels_disc[randomize_disc]

        print("Train Discriminator on batches of size: ", params.DISC_BATCH_SIZE)

        # train
        history = disc.train(queries_disc, documents_disc, labels_disc)

        # Plot the progress
        for i, epoch_disc in enumerate(history.epoch):
            print("Epoch %s [G loss: %f, acc.: %.2f%%]" % (epoch_disc, history.history["loss"][i], history.history["acc"][i]))
            step = int(epoch * 10 + epoch_disc)
            experiment.log_metric('loss_train_disc', history.history["loss"][i], step=step)
            experiment.log_metric('accuracy_train_disc', history.history["acc"][i], step=step)

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

        last_disc = disc
        last_gen = gen

        best_disc, best_gen, p_best_val, ndcg_best_val = __get_best_eval_result(disc, best_disc, gen, best_gen, p_step,
                                                                                p_best_val, ndcg_step, ndcg_best_val)

    print("Best:", "disc p@5 =", p_best_val, "disc ndcg@5 =", ndcg_best_val)

    return best_gen, best_disc, p_best_val, ndcg_best_val


def __build_train_data(x_train, ratings_data, queries_data):
    train_queries_data = {}
    train_ratings_data = {}

    for query_id in x_train:
        try:
            train_ratings_data[query_id] = ratings_data[query_id]
            train_queries_data[query_id] = queries_data[query_id]
        except KeyError:
            print("No rating data exist for query-id:", query_id)

    return train_ratings_data, train_queries_data


def __get_query_specific_data(query_id, ratings_data, documents_data):
    # get all query specific ratings
    if params.DATA_SOURCE == params.DATA_SOURCE_TREC_CDS_2017:
        x_pos_relevant = []
        for i, rating in enumerate(ratings_data[query_id]):
            if ratings_data[query_id][rating] > 0:
                x_pos_relevant.append(rating)

        x_pos_list = list(x_pos_relevant)[:params.POS_TRAINING_DATA_PER_QUERY]
    else:
        x_pos_list = list(ratings_data[query_id].keys())[:params.POS_TRAINING_DATA_PER_QUERY]

    # get all other ratings
    docs_pos_ids = np.unique(x_pos_list)
    candidate_list = []
    for doc_id in documents_data.keys():
        if doc_id not in docs_pos_ids:
            candidate_list.append(doc_id)

    return x_pos_list, candidate_list


def __generate_negatives_for_discriminator_pretrain(x_train, ratings_data, documents_data):
    pos = []
    neg = []

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        if params.DATA_SOURCE == params.DATA_SOURCE_TREC_CDS_2017:
            neg_list = __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, 3 * len(x_pos_list))
        else:
            doc_ids = __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, 3 * len(x_pos_list))
            neg_list = np.random.choice(doc_ids, size=[len(x_pos_list)])

        for i in range(len(x_pos_list)):
            pos.append([query_id, x_pos_list[i]])
            neg.append([query_id, neg_list[i]])

    return pos, neg


def __generate_negatives_for_discriminator(gen, x_train, ratings_data, queries_data, documents_data):
    pos = []
    neg = []

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

        neg_data_queries.extend(cand_queries)

        neg_data_documents.extend(cand_documents)

        neg_data_q_ids.extend(cand_q_ids)

    # importance Sampling
    probs = gen.get_prob(neg_data_queries, neg_data_documents)
    print("__get_rand_batch_from_candidates_for_negatives: prob = ", probs)

    neg_data_cands = {}
    for i, prob in enumerate(probs):
        query_id = neg_data_q_ids[i]
        if query_id in neg_data_cands.keys():
            neg_data_cands[query_id] = np.append(neg_data_cands[query_id], [prob])
        else:
            neg_data_cands[query_id] = [prob]

    for query_id, neg_data_probs in neg_data_cands.items():

        probs_rand = neg_data_probs[:]
        probs_rand /= probs_rand.sum().astype(float)

        if params.DATA_SOURCE == params.DATA_SOURCE_TREC_CDS_2017:
            neg_list = np.asarray(neg_data[query_id])
        else:
            neg_list = np.random.choice(neg_data[query_id], size=[len(pos_data[query_id])], p=probs_rand)

        neg_data[query_id] = neg_list

    for query_id, pos_values in pos_data.items():
        for i, pos_elem in enumerate(pos_values):
            neg_elem = neg_data[query_id][i]

            pos.append([query_id, pos_elem])
            neg.append([query_id, neg_elem])

    return pos, neg


def __generate_negatives_for_generator(gen, x_train, ratings_data, queries_data, documents_data):
    data = []
    data_probs_is = []

    pos_neg_data = {}
    pos_data = {}

    pos_neg_data_q_ids = []
    pos_neg_data_d_ids = []
    pos_neg_data_queries = []
    pos_neg_data_documents = []

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        x_pos_list, candidate_list = __get_query_specific_data(query_id, ratings_data, documents_data)

        candidate_ids = __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, 2 * len(x_pos_list))

        pos_neg_data[query_id] = np.append(x_pos_list, candidate_ids)
        pos_data[query_id] = np.asarray(x_pos_list)

        # prepare neg data
        cand_queries = [queries_data[query_id]] * len(pos_neg_data[query_id])
        cand_documents = [documents_data[x] for x in pos_neg_data[query_id]]
        cand_q_ids = [query_id] * len(pos_neg_data[query_id])
        cand_d_ids = [x for x in pos_neg_data[query_id]]

        pos_neg_data_queries.extend(cand_queries)
        pos_neg_data_documents.extend(cand_documents)
        pos_neg_data_q_ids.extend(cand_q_ids)
        pos_neg_data_d_ids.extend(cand_d_ids)

    pos_neg_data_queries = np.asarray(pos_neg_data_queries)
    pos_neg_data_documents = np.asarray(pos_neg_data_documents)
    pos_neg_data_q_ids = np.asarray(pos_neg_data_q_ids)
    pos_neg_data_d_ids = np.asarray(pos_neg_data_d_ids)

    # importance sampling
    probs = gen.get_prob(pos_neg_data_queries, pos_neg_data_documents)
    print("__generate_negatives_for_generator: prob = ", probs)

    exp_rating = np.exp(probs - np.max(probs))
    probs = exp_rating / np.sum(exp_rating)

    # importance sampling, change doc prob
    probs_is = probs * (1.0 - params.GEN_LAMBDA)

    for i, query_id in enumerate(pos_neg_data_q_ids):
        if pos_neg_data_d_ids[i] in pos_data[query_id]:
            probs_is[i] += (params.GEN_LAMBDA / (1.0 * len(pos_data[query_id])))

    data_cands_prob = {}
    data_cands_prob_is = {}
    for i, query_id in enumerate(pos_neg_data_q_ids):
        if query_id in data_cands_prob.keys():
            data_cands_prob[query_id] = np.append(data_cands_prob[query_id], [probs[i]])
            data_cands_prob_is[query_id] = np.append(data_cands_prob_is[query_id], [probs_is[i]])
        else:
            data_cands_prob[query_id] = [probs[i]]
            data_cands_prob_is[query_id] = [probs_is[i]]

    for query_id, probs in data_cands_prob_is.items():

        probs_rand = probs[:]
        probs_rand /= probs_rand.sum().astype(float)

        choosen_indexes = np.random.choice(np.arange(len(pos_neg_data[query_id])), size=[2 * len(pos_data[query_id])], p=probs_rand)
        choosen_data = pos_neg_data[query_id][choosen_indexes]
        # prob / important sampling prob (loss => prob * reward * prob / important sampling prob)
        choosen_data_prob_is = np.array(data_cands_prob[query_id])[choosen_indexes] / np.array(data_cands_prob_is[query_id])[choosen_indexes]

        for i, doc_id in enumerate(choosen_data):
            data.append([query_id, doc_id])
            data_probs_is.append(choosen_data_prob_is[i])

    return data, data_probs_is


def __get_rand_batch_from_candidates_for_negatives(query_id, candidate_list, size):
    # create ES client, create index
    es = Elasticsearch(hosts=[params.ES_HOST])

    if params.DATA_SOURCE == params.DATA_SOURCE_TREC_CDS_2017:
        query_text = es.get(index="queries_trec", doc_type="query", id=str(query_id))["_source"]["text"]
        query_text = query_text.replace("\n", " ")
        query_text = query_text.replace("\"", " ")
        query_text = query_text[0:9999]
        candidates = es.search(index="documents_trec", body={"query": {"match": {"text": query_text}}}, size=size)
        candidates = [doc['_id'] for doc in candidates['hits']['hits']]
    else:
        query_text = es.get(index="queries", doc_type="query", id=str(query_id))["_source"]["text"]
        query_text = query_text.replace("\n", " ")
        query_text = query_text.replace("\"", " ")
        query_text = query_text[0:9999]
        candidates = es.search(index="documents", body={"query": {"match": {"text": query_text}}}, size=size)
        candidates = [doc['_id'] for doc in candidates['hits']['hits']]

    # prepare pos and neg data
    doc_ids = np.array(candidates)

    if len(candidates) < size:
        candidates_addition = np.random.choice(candidate_list, size=[size - len(candidates)])
        doc_ids = np.append(doc_ids, candidates_addition)

    return doc_ids


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
