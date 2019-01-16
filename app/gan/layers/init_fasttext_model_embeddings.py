import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import fasttext

import parameters as params

# based on: https://pypi.org/project/fasttext/


def load_model():
    print('Loading fastText embedding model from', params.FASTTEXT)
    model = fasttext.load_model(params.FASTTEXT)
    return model


def __build_embeddings_matrix(tokenizer: Tokenizer, model, max_num_words):
    print('Preparing embedding matrix.')
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    embeddings_matrix = np.zeros((num_words, params.EMBEDDING_DIM))

    for word, i in word_index.items():
        if i == 0:
            raise ValueError("index 0 is not allowed in embedding matrix")
        elif i > max_num_words:
            continue

        try:
            embedding_vector = model[word]
        except Exception:
            raise ValueError("no embedding vector found for word", word)

        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    return num_words, embeddings_matrix


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
def __build_embeddings_layer(num_words, embeddings_matrix, max_sequence_length):
    embeddings_layer = Embedding(input_dim=num_words,
                                 output_dim=params.EMBEDDING_DIM,
                                 weights=[embeddings_matrix],
                                 input_length=max_sequence_length,
                                 mask_zero=True,
                                 trainable=False)
    return embeddings_layer


def init_embedding_layer(tokenizer, model, max_sequence_length, max_num_words):
    num_words, embeddings_matrix = __build_embeddings_matrix(tokenizer, model, max_num_words)
    embeddings_layer = __build_embeddings_layer(num_words, embeddings_matrix, max_sequence_length)
    return embeddings_layer
