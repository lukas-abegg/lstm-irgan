from keras import backend as K
from keras.layers import Bidirectional, Embedding, GRU, Dense, Activation, Lambda, Concatenate
from keras.layers.core import Reshape, Dropout
from keras.models import Model, Input, save_model, load_model

from app.gan.optimizer.AdamW import AdamW

import numpy as np
import app.parameters as params


class Generator:
    def __init__(self, samples_per_epoch=0, weight_decay=None, learning_rate=None, temperature=1.0, dropout=0.2, embedding_layer_q=None, embedding_layer_d=None, model=None, sess=None):
        self.weight_decay = weight_decay
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.dropout = dropout
        self.embeddings_layer_q: Embedding = embedding_layer_q
        self.embeddings_layer_d: Embedding = embedding_layer_d
        self.model: Model = self.__get_model(model)
        self.sess = sess

    def __get_model(self, model):
        if model is None:
            return self.__init_model()
        else:
            return model

    def __init_model(self):
        # create model
        reward = Input(shape=(None,), name='input_reward')
        important_sampling = Input(shape=(None,), name='input_imp_sampling')

        sequence_input_q = Input(shape=(params.MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_query')
        embedded_sequences_q = self.embeddings_layer_q(sequence_input_q)

        lstm_q_in = Bidirectional(GRU(params.GEN_HIDDEN_SIZE_LSTM, return_sequences=True, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(embedded_sequences_q)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        lstm_q_out = Bidirectional(GRU(params.GEN_HIDDEN_SIZE_LSTM, return_sequences=False, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(lstm_q_in)

        sequence_input_d = Input(shape=(params.MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_doc')
        embedded_sequences_d = self.embeddings_layer_d(sequence_input_d)

        lstm_d_in = Bidirectional(GRU(params.GEN_HIDDEN_SIZE_LSTM, return_sequences=True, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(embedded_sequences_d)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        lstm_d_out = Bidirectional(GRU(params.GEN_HIDDEN_SIZE_LSTM, return_sequences=False, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(lstm_d_in)

        x = Concatenate()([lstm_q_out, lstm_d_out])
        x = Dropout(self.dropout)(x)

        # we stack a deep fully-connected network on top
        x = Dense(params.GEN_HIDDEN_SIZE_DENSE,
                  activation='elu')(x)
        x = Dense(1, activation='elu')(x)

        score = Lambda(lambda z: z / self.temperature, name='raw_score')(x)
        score = Reshape([-1], name='score')(score)
        prob = Activation('softmax', name='prob')(score)

        model = Model(inputs=[sequence_input_q, sequence_input_d, reward, important_sampling], outputs=[prob])
        model.summary()

        adamw = AdamW(batch_size=params.GEN_BATCH_SIZE, samples_per_epoch=self.samples_per_epoch, epochs=params.GEN_TRAIN_EPOCHS)

        model.compile(loss=self.__loss(reward, important_sampling),
                      optimizer=adamw,
                      metrics=['accuracy'])

        return model

    @staticmethod
    def __loss(_reward, _important_sampling):
        def _loss(y_true, y_pred):
            log_action_prob = K.log(y_pred)
            loss = - K.reshape(log_action_prob, [-1]) * K.reshape(_reward, [-1]) * K.reshape(_important_sampling, [-1])
            loss = K.mean(loss)
            return loss

        return _loss

    def train(self, train_data_queries, train_data_documents, reward, important_sampling):
        self.model.train_on_batch([train_data_queries, train_data_documents, reward, important_sampling], np.zeros([train_data_queries.shape[0]]))

    def get_score(self, train_data_queries, train_data_documents):
        inputs = self.model.inputs + [K.learning_phase()]
        out = self.model.get_layer('score').output
        functor = K.function(inputs, [out])
        layer_outs = functor([train_data_queries, train_data_documents, 0.])
        return layer_outs

    def get_prob(self, train_data_queries, train_data_documents):
        inputs = self.model.inputs + [K.learning_phase()]
        out = self.model.get_layer('prob').output
        functor = K.function(inputs, [out])
        layer_outs = functor([train_data_queries, train_data_documents, 0.])
        return layer_outs

    def save_model(self, filepath):
        save_model(self.model, filepath)
        print("Saved model to disk")

    @staticmethod
    def load_model(filepath):
        model = load_model(filepath)
        print("Loaded model from disk")

        gen = Generator(model=model)
        return gen

    @staticmethod
    def create_model(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q, embedding_layer_d, sess):

        gen = Generator(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q, embedding_layer_d, sess=sess)
        return gen
