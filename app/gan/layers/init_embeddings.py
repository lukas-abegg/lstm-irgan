import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.initializers import Constant

import app.parameters as params


# based on: https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py

def __build_index_mapping():
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(params.WORD2VEC) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def __build_embeddings_matrix(tokenizer: Tokenizer, embeddings_index):
    print('Preparing embedding matrix.')
    word_index = tokenizer.word_index
    num_words = min(params.EMBEDDINGS_MAX_NUM_WORDS, len(word_index)) + 1
    embeddings_matrix = np.zeros((num_words, params.EMBEDDINGS_DIM))

    for word, i in word_index.items():
        if i > params.EMBEDDINGS_MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
    return num_words, embeddings_matrix


def __build_embeddings_layer(num_words, embeddings_matrix, max_sequence_length):
    embeddings_layer = Embedding(num_words,
                                 params.EMBEDDINGS_MAX_NUM_WORDS,
                                 embeddings_initializer=Constant(embeddings_matrix),
                                 input_length=max_sequence_length,
                                 trainable=False)
    return embeddings_layer


def init_embedding_layer(tokenizer: Tokenizer, max_sequence_length):
    embeddings_index = __build_index_mapping()
    (num_words, embeddings_matrix) = __build_embeddings_matrix(tokenizer, embeddings_index)
    embeddings_layer = __build_embeddings_layer(num_words, embeddings_matrix, max_sequence_length)
    return embeddings_layer
