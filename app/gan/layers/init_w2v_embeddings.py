import numpy as np
from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding

import parameters as params


# based on: https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py

# first, build index mapping words in the embeddings set
# to their embedding vector
def build_index_mapping():
    print('Indexing word vectors.')
    embeddings_index = KeyedVectors.load_word2vec_format(params.FASTTEXT, binary=params.FASTTEXT_BINARY)
    print('Found %s word vectors.' % len(embeddings_index.vocab))
    return embeddings_index


def __build_embeddings_matrix(tokenizer: Tokenizer, embeddings_index):
    print('Preparing embedding matrix.')
    word_index = tokenizer.word_index
    num_words = min(params.MAX_NUM_WORDS, len(word_index)) + 1
    embeddings_matrix = np.zeros((num_words, params.EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > params.MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
    return num_words, embeddings_matrix


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
def __build_embeddings_layer(num_words, embeddings_matrix, max_sequence_length):
    embeddings_layer = Embedding(input_dim=num_words,
                                 output_dim=params.EMBEDDING_DIM,
                                 weights=[embeddings_matrix],
                                 input_length=max_sequence_length,
                                 trainable=False)
    return embeddings_layer


def init_embedding_layer(tokenizer, embeddings_index, max_sequence_length):
    num_words, embeddings_matrix = __build_embeddings_matrix(tokenizer, embeddings_index)
    embeddings_layer = __build_embeddings_layer(num_words, embeddings_matrix, max_sequence_length)
    return embeddings_layer