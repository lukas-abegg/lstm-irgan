import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding

import fasttext

import app.parameters as params

# based on: https://pypi.org/project/fasttext/


def load_model():
    print('load fasttext model from', params.FASTTEXT)
    model = fasttext.load_model(params.FASTTEXT)
    return model


def __build_embeddings_matrix(tokenizer: Tokenizer, model):
    print('Preparing embedding matrix.')
    word_index = tokenizer.word_index
    num_words = min(params.MAX_NUM_WORDS, len(word_index)) + 1
    embeddings_matrix = np.zeros((num_words, params.EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > params.MAX_NUM_WORDS:
            continue
        embedding_vector = model[word]
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


def init_embedding_layer(tokenizer, model, max_sequence_length):
    num_words, embeddings_matrix = __build_embeddings_matrix(tokenizer, model)
    embeddings_layer = __build_embeddings_layer(num_words, embeddings_matrix, max_sequence_length)
    return embeddings_layer
