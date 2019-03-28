import os
import warnings

from comet_ml import Experiment
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import numpy as np

import tensorflow as tf
from keras import backend

from keras.layers import Dense, Bidirectional, Embedding, GRU, Concatenate
from keras.layers.core import Dropout
from keras.models import Model, Input
from keras import regularizers
from keras.optimizers import RMSprop, SGD, Adam

import parameters as params

import fastText as fasttext


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


def data():
    from sklearn.model_selection import train_test_split

    queries = np.load('../hyperas/queries_hyperas.npy')
    documents = np.load('../hyperas/documents_hyperas.npy')
    labels = np.load('../hyperas/labels_hyperas.npy')

    ids = np.arange(len(queries))

    X_train, X_test = train_test_split(ids, test_size=0.10, random_state=777)
    X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=777)

    X_train_queries = queries[X_train]
    X_test_queries = queries[X_test]
    X_val_queries = queries[X_val]

    X_train_documents = documents[X_train]
    X_test_documents = documents[X_test]
    X_val_documents = documents[X_val]

    X_train_data = []
    for i in np.arange(len(X_train_queries)):
        X_train_data.append((X_train_queries[i], X_train_documents[i]))

    X_test_data = []
    for i in np.arange(len(X_test_queries)):
        X_test_data.append((X_test_queries[i], X_test_documents[i]))

    X_val_data = []
    for i in np.arange(len(X_val_queries)):
        X_val_data.append((X_val_queries[i], X_val_documents[i]))

    y_train = labels[X_train]
    y_test = labels[X_test]
    y_val = labels[X_val]

    return X_train_data, X_val_data, X_test_data, y_train, y_val, y_test


def model(X_train_data, X_val_data, X_test_data, y_train, y_val, y_test):

    # -----------------------------------------------
    # Prepare X-Data
    # -----------------------------------------------
    X_train_queries = []
    X_train_documents = []
    for x in X_train:
        X_train_queries.append(x[0])
        X_train_documents.append(x[1])

    X_train_queries = np.asarray(X_train_queries)
    X_train_documents = np.asarray(X_train_documents)

    X_val_queries = []
    X_val_documents = []
    for x in X_val:
        X_val_queries.append(x[0])
        X_val_documents.append(x[1])

    X_val_queries = np.asarray(X_val_queries)
    X_val_documents = np.asarray(X_val_documents)

    # -----------------------------------------------
    # Tokenizer
    # -----------------------------------------------
    # Load
    word_index = np.load('../hyperas/word_index.npy').item()

    # -----------------------------------------------
    # Embedding-Layers
    # -----------------------------------------------
    # print('Loading fastText embedding model from', params.FASTTEXT)
    # model = fasttext.load_model(params.FASTTEXT)
    #
    # print('Preparing embedding matrix for queries.')
    # word_index = word_index
    # num_words = len(word_index) + 1
    # embeddings_matrix = np.zeros((num_words, params.EMBEDDING_DIM))
    #
    # for word, i in word_index.items():
    #     if i == 0:
    #         raise ValueError("index 0 is not allowed in embedding matrix")
    #     elif i > params.MAX_NUM_WORDS_QUERIES:
    #         continue
    #
    #     try:
    #         embedding_vector = model.get_word_vector(word)
    #     except Exception:
    #         raise ValueError("no embedding vector found for word", word)
    #
    #     if embedding_vector is not None:
    #         embeddings_matrix[i] = embedding_vector
    #
    # print('Create embedding layer for queries')
    # embedding_layer_q = Embedding(input_dim=num_words,
    #                               output_dim=params.EMBEDDING_DIM,
    #                               weights=[embeddings_matrix],
    #                               input_length=params.MAX_SEQUENCE_LENGTH_QUERIES,
    #                               mask_zero=True,
    #                               trainable=False)
    #
    # print('Preparing embedding matrix for queries.')
    # word_index = word_index
    # num_words = len(word_index) + 1
    # embeddings_matrix = np.zeros((num_words, params.EMBEDDING_DIM))
    #
    # for word, i in word_index.items():
    #     if i == 0:
    #         raise ValueError("index 0 is not allowed in embedding matrix")
    #     elif i > params.MAX_NUM_WORDS_DOCS:
    #         continue
    #
    #     try:
    #         embedding_vector = model.get_word_vector(word)
    #     except Exception:
    #         raise ValueError("no embedding vector found for word", word)
    #
    #     if embedding_vector is not None:
    #         embeddings_matrix[i] = embedding_vector
    #
    # print('Create embedding layer for queries')
    # embedding_layer_d = Embedding(input_dim=num_words,
    #                               output_dim=params.EMBEDDING_DIM,
    #                               weights=[embeddings_matrix],
    #                               input_length=params.MAX_SEQUENCE_LENGTH_DOCS,
    #                               mask_zero=True,
    #                               trainable=False)

    embedding_layer_q = Embedding(input_dim=len(word_index) + 1,
                                  output_dim=params.EMBEDDING_DIM,
                                  weights=None,
                                  input_length=params.MAX_SEQUENCE_LENGTH_QUERIES,
                                  mask_zero=True,
                                  trainable=False)

    embedding_layer_d = Embedding(input_dim=len(word_index) + 1,
                                  output_dim=params.EMBEDDING_DIM,
                                  weights=None,
                                  input_length=params.MAX_SEQUENCE_LENGTH_DOCS,
                                  mask_zero=True,
                                  trainable=False)
    # -----------------------------------------------
    # Model
    # -----------------------------------------------

    #weight_decay = {{uniform(params.OPT_MIN_WEIGHT_DECAY, params.OPT_MAX_WEIGHT_DECAY)}}
    weight_decay = 0.2
    #dropout = {{uniform(params.OPT_MIN_DROPOUT, params.OPT_MAX_DROPOUT)}}
    dropout = 0.2

    # create model
    sequence_input_q = Input(shape=(params.MAX_SEQUENCE_LENGTH_QUERIES,), dtype='int32', name='input_query')
    embedded_sequences_q = embedding_layer_q(sequence_input_q)

    lstm_q_in = Bidirectional(
        GRU(64, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', return_sequences=True, activation='elu',
            dropout=dropout, recurrent_dropout=dropout))(embedded_sequences_q)
    # this LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_q_out = Bidirectional(
        GRU(64, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', return_sequences=False, activation='elu',
            dropout=dropout, recurrent_dropout=dropout))(lstm_q_in)

    sequence_input_d = Input(shape=(params.MAX_SEQUENCE_LENGTH_DOCS,), dtype='int32', name='input_doc')
    embedded_sequences_d = embedding_layer_d(sequence_input_d)

    lstm_d_in = Bidirectional(
        GRU(64, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', return_sequences=True, activation='elu',
            dropout=dropout, recurrent_dropout=dropout))(embedded_sequences_d)
    # this LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_d_out = Bidirectional(
        GRU(64, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', return_sequences=False, activation='elu',
            dropout=dropout, recurrent_dropout=dropout))(lstm_d_in)

    x = Concatenate()([lstm_q_out, lstm_d_out])
    x = Dropout(dropout)(x)

    x = Dense(46, activation='elu', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform')(x)
    prob = Dense(1, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer='random_uniform', activation='sigmoid', name='prob')(x)

    model = Model(inputs=[sequence_input_q, sequence_input_d], outputs=[prob])

    model.compile(loss='binary_crossentropy',
                  #optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  optimizer='adam',
                  metrics=['accuracy'])

    from keras import callbacks

    reduce_lr = callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit({'sequence_input_q': X_train_queries, 'sequence_input_d': X_train_documents},
              {'prob': y_train},
              epochs=5,
              #batch_size={{choice([100, 150, 200])}},
              batch_size=200,
              validation_data=({'sequence_input_q': X_val_queries, 'sequence_input_d': X_val_documents},
                               {'prob': y_val}),
              callbacks=[reduce_lr])

    score, acc = model.evaluate({'sequence_input_q': X_val_queries, 'sequence_input_d': X_val_documents},
                                {'prob': y_val},
                                batch_size=200,
                                verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(graph=tf.get_default_graph(), config=tf_config)

    backend.set_session(sess)

    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # experiment = Experiment(api_key="tgrD5ElfTdvaGEmJB7AEZG8Ra",
    #                         project_name="general", workspace="abeggluk")

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    X_train, X_val, X_test, y_train, y_val, y_test = data()

    X_test_queries = []
    X_test_documents = []
    for x in X_test:
        X_test_queries.append(x[0])
        X_test_documents.append(x[1])

    X_test_queries = np.asarray(X_test_queries)
    X_test_documents = np.asarray(X_test_documents)

    print("Evalutation of best performing model:")
    print(best_model.evaluate([X_test_queries, X_test_documents], y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

