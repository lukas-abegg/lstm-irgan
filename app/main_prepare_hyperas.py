import os
import warnings

import tensorflow as tf
from keras import backend

from elasticsearch import Elasticsearch

import data_preparation.init_data_example as init_example
import data_preparation.init_data_wikiclir as init_wikiclir
import data_preparation.init_data_nfcorpus as init_nfcorpus
import parameters as params

import numpy as np

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
    x_train = query_ids
    return sess, x_train, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d


def create_eval_data(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess):
    x_train = np.array(x_train)

    # -----------------------------------------------
    # build_train_data
    # -----------------------------------------------

    train_queries_data = {}
    train_ratings_data = {}

    for query_id in x_train:
        train_ratings_data[query_id] = ratings_data[query_id]
        train_queries_data[query_id] = queries_data[query_id]

    # -----------------------------------------------
    # generate_negatives
    # -----------------------------------------------

    pos = []
    neg = []

    print('start negative sampling for discriminator using generator')
    for query_id in x_train:
        # get query specific rating and all relevant docs
        # get all query specific ratings
        x_pos_list = list(ratings_data[query_id].keys())[:params.POS_TRAINING_DATA_PER_QUERY]

        # get all other ratings
        docs_pos_ids = np.unique(x_pos_list)
        candidate_list = []
        for doc_id in documents_data.keys():
            if doc_id not in docs_pos_ids:
                candidate_list.append(doc_id)

        # -----------------------------------------------
        # get_rand_batch_from_candidates_for_negatives
        # -----------------------------------------------

        size = 3 * len(x_pos_list)
        # create ES client, create index
        es = Elasticsearch(hosts=[params.ES_HOST])
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

        neg_list = np.random.choice(doc_ids, size=[len(x_pos_list)])

        for i in range(len(x_pos_list)):
            pos.append([query_id, x_pos_list[i]])
            neg.append([query_id, neg_list[i]])

    # -----------------------------------------------
    # prepare pos and neg data
    # -----------------------------------------------

    pos_data_queries = [queries_data[x[0]] for x in pos]
    pos_data_documents_disc = [documents_data[x[1]] for x in pos]
    neg_data_queries = [queries_data[x[0]] for x in neg]
    neg_data_documents_disc = [documents_data[x[1]] for x in neg]

    queries = pos_data_queries[:]
    queries.extend(neg_data_queries)
    queries = np.asarray(queries)

    documents = pos_data_documents_disc[:]
    documents.extend(neg_data_documents_disc)
    documents = np.asarray(documents)

    # prepare pos and neg label
    pos_data_label_disc = [1.0] * len(pos_data_queries)
    neg_data_label_disc = [0.0] * len(neg_data_queries)

    labels = pos_data_label_disc[:]
    labels.extend(neg_data_label_disc)
    labels = np.asarray(labels)

    randomize = np.arange(len(queries))
    np.random.shuffle(randomize)

    queries = queries[randomize]
    documents = documents[randomize]
    labels = labels[randomize]

    np.save("../hyperas/queries_hyperas.npy", queries)
    np.save("../hyperas/documents_hyperas.npy", documents)
    np.save("../hyperas/labels_hyperas.npy", labels)

    np.save('../hyperas/word_index.npy', tokenizer_q.word_index)


def main():
    sess, x_train, ratings_data, documents_data, queries_data, tokenizer_q, tokenizer_d = get_env_data_with_x_data_splitted()
    create_eval_data(x_train, ratings_data, queries_data, documents_data, tokenizer_q, tokenizer_d, sess)


if __name__ == '__main__':
    main()
