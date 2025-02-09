from keras import regularizers
from keras.layers import Bidirectional, Embedding, GRU, Dense, Activation, Concatenate
from keras.layers.core import Dropout
from keras.models import Model, Input, load_model, model_from_json

from gan.optimizer.AdamW import AdamW

import numpy as np
import parameters as params


class GeneratorPretrain:
    def __init__(self, samples_per_epoch=0, weight_decay=None, learning_rate=None, temperature=1.0, dropout=0.2,
                 embedding_layer_q=None, embedding_layer_d=None, model=None, sess=None):
        self.weight_decay = weight_decay
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.dropout = dropout
        self.embeddings_layer_q: Embedding = embedding_layer_q
        self.embeddings_layer_d: Embedding = embedding_layer_d
        self.reward = Input(shape=(None,), name='input_reward')
        self.important_sampling = Input(shape=(None,), name='input_imp_sampling')
        self.adamw = AdamW(lr=self.learning_rate, batch_size=params.GEN_BATCH_SIZE,
                           samples_per_epoch=self.samples_per_epoch, epochs=params.GEN_TRAIN_EPOCHS)
        self.sess = sess
        self.__get_model(model)

    def __get_model(self, model):
        if model is None:
            self.__init_model()
        else:
            self.model = model

    def __init_model(self):
        # create model

        self.sequence_input_q = Input(shape=(params.MAX_SEQUENCE_LENGTH_QUERIES,), dtype='int32', name='input_query')
        self.embedded_sequences_q = self.embeddings_layer_q(self.sequence_input_q)

        self.lstm_q_in = Bidirectional(
            GRU(params.GEN_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=True, activation='elu', dropout=self.dropout,
                recurrent_dropout=self.dropout))(self.embedded_sequences_q)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        self.lstm_q_out = Bidirectional(
            GRU(params.GEN_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=False, activation='elu', dropout=self.dropout,
                recurrent_dropout=self.dropout))(self.lstm_q_in)

        self.sequence_input_d = Input(shape=(params.MAX_SEQUENCE_LENGTH_DOCS,), dtype='int32', name='input_doc')
        self.embedded_sequences_d = self.embeddings_layer_d(self.sequence_input_d)

        self.lstm_d_in = Bidirectional(
            GRU(params.GEN_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=True, activation='elu', dropout=self.dropout,
                recurrent_dropout=self.dropout))(self.embedded_sequences_d)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        self.lstm_d_out = Bidirectional(
            GRU(params.GEN_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=False, activation='elu', dropout=self.dropout,
                recurrent_dropout=self.dropout))(self.lstm_d_in)

        self.x = Concatenate()([self.lstm_q_out, self.lstm_d_out])
        self.x = Dropout(self.dropout)(self.x)

        # we stack a deep fully-connected network on top
        self.x = Dense(params.GEN_HIDDEN_SIZE_DENSE, activation='elu', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='random_uniform')(self.x)
        self.x = Dense(2, kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='random_uniform')(self.x)

        self.prob = Activation('softmax', name='prob')(self.x)

        self.model = Model(inputs=[self.sequence_input_q, self.sequence_input_d, self.reward, self.important_sampling],
                           outputs=[self.prob])

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.adamw,
                           metrics=['accuracy'])

    def train(self, train_data_queries, train_data_documents, train_data_label):
        input_reward = [0.0] * len(train_data_queries)
        input_reward = np.asarray(input_reward)
        input_important_sampling = [0.0] * len(train_data_queries)
        input_important_sampling = np.asarray(input_important_sampling)
        return self.model.train_on_batch([train_data_queries, train_data_documents, input_reward, input_important_sampling], train_data_label)

    def get_prob(self, train_data_queries, train_data_documents):
        input_reward = [0.0] * len(train_data_queries)
        input_reward = np.asarray(input_reward)
        input_important_sampling = [0.0] * len(train_data_queries)
        input_important_sampling = np.asarray(input_important_sampling)
        pred_scores = self.model.predict(
            [train_data_queries, train_data_documents, input_reward, input_important_sampling], params.GEN_BATCH_SIZE)
        return pred_scores[:, 1]

    def save_model_to_file(self, filepath):
        self.model.save(filepath)
        print("Saved model to disk")

    def save_model_to_weights(self, filepath_json, filepath_weights):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(filepath_json, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(filepath_weights)
        print("Saved model weights to disk")

    @staticmethod
    def load_model_from_file(filepath):
        loaded_model = load_model(filepath)
        print("Loaded model from disk")

        gen = GeneratorPretrain(model=loaded_model)
        gen.model.compile(loss='categorical_crossentropy',
                          optimizer=gen.adamw,
                          metrics=['accuracy'])
        return gen

    @staticmethod
    def load_model_from_weights(filepath_json, filepath_weights):
        # load json and create model
        json_file = open(filepath_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(filepath_weights)
        print("Loaded model from disk")

        gen = GeneratorPretrain(model=loaded_model)
        gen.model.compile(loss='categorical_crossentropy',
                          optimizer=gen.adamw,
                          metrics=['accuracy'])
        return gen

    @staticmethod
    def create_model(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q,
                     embedding_layer_d, sess):

        gen = GeneratorPretrain(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q,
                        embedding_layer_d, sess=sess)
        return gen
