import os
import tensorflow as tf
from keras import initializers
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Activation, Dropout
from keras.layers import Bidirectional, Embedding, LSTM
from keras import backend as k
from keras import regularizers
from keras import optimizers
from keras_preprocessing.text import Tokenizer

from app.gan.layers.init_embeddings import init_embedding_layer
import app.parameters as params


class Discriminator:
    def __init__(self, feature_size, hidden_size, weight_decay, learning_rate):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        tokenizer = Tokenizer(num_words=params.EMBEDDINGS_MAX_NUM_WORDS)
        tokenizer.fit_on_texts(texts)

        embeddings_layer = init_embedding_layer(tokenizer, params.EMBEDDINGS_MAX_SEQ_LENGTH)

        # create model
        model = Sequential()
        model.add(embeddings_layer)
        model.add(Bidirectional(LSTM(64)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, train_data, train_data_label):
        self.model.train_on_batch(train_data, train_data_label)

    def save_model(self, filename):
        self.model.save_weights(filename)

    @staticmethod
    def create_model(feature_size):
        # call discriminator, generator
        disc = Discriminator(feature_size, params.DISC_HIDDEN_SIZE, params.DISC_WEIGHT_DECAY,
                             params.DISC_LEARNING_RATE)
        return disc
