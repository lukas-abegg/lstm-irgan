from keras import backend as K
from keras.layers import Dense, Activation, Bidirectional, Embedding, GRU, Concatenate
from keras.layers.core import Reshape, Dropout
from keras.models import Model, Input, save_model, load_model

from app.gan.optimizer.AdamW import AdamW

import app.parameters as params


class Discriminator:
    def __init__(self, samples_per_epoch=0, weight_decay=None, learning_rate=None, dropout=0.2, embedding_layer_q=None, embedding_layer_d=None, model=None):
        self.weight_decay = weight_decay
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.embeddings_layer_q: Embedding = embedding_layer_q
        self.embeddings_layer_d: Embedding = embedding_layer_d
        self.model: Model = self.__get_model(model)

    def __get_model(self, model):
        if model is None:
            return self.__init_model()
        else:
            return model

    def __init_model(self):
        # create model
        sequence_input_q = Input(shape=(params.MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_query')
        embedded_sequences_q = self.embeddings_layer_q(sequence_input_q)

        lstm_q_in = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, return_sequences=True, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(embedded_sequences_q)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        lstm_q_out = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, return_sequences=False, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(lstm_q_in)

        sequence_input_d = Input(shape=(params.MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_doc')
        embedded_sequences_d = self.embeddings_layer_d(sequence_input_d)

        lstm_d_in = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, return_sequences=True, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(embedded_sequences_d)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        lstm_d_out = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, return_sequences=False, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(lstm_d_in)

        x = Concatenate()([lstm_q_out, lstm_d_out])
        x = Dropout(self.dropout)(x)

        x = Dense(params.DISC_HIDDEN_SIZE_DENSE,
                  activation='elu')(x)
        x = Dense(1, activation='elu')(x)

        score = Reshape([-1])(x)
        prob = Activation('sigmoid', name='prob')(score)

        model = Model(inputs=[sequence_input_q, sequence_input_d], outputs=[prob])
        model.summary()

        adamw = AdamW(batch_size=params.DISC_BATCH_SIZE, samples_per_epoch=self.samples_per_epoch,
                      epochs=params.DISC_TRAIN_EPOCHS)

        model.compile(loss='binary_crossentropy',
                      optimizer=adamw,
                      metrics=['accuracy'])

        return model

    def train(self, train_data_queries, train_data_documents, train_data_label):
        self.model.train_on_batch([train_data_queries, train_data_documents], train_data_label)

    def get_preresult(self, train_data_queries, train_data_documents):
        return (self.model.predict([train_data_queries, train_data_documents]) - 0.5) * 2

    def get_reward(self, train_data_queries, train_data_documents):
        inputs = self.model.inputs + [K.learning_phase()]
        out = self.model.get_layer("prob").output
        functor = K.function(inputs, [out])
        layer_outs = functor([train_data_queries, train_data_documents, 1.])
        return (layer_outs[0] - 0.5) * 2

    def save_model(self, filepath):
        save_model(self.model, filepath)
        print("Saved model to disk")

    @staticmethod
    def load_model(filepath):
        model = load_model(filepath)
        print("Loaded model from disk")

        disc = Discriminator(model=model)
        return disc

    @staticmethod
    def create_model(samples_per_epoch, weight_decay, learning_rate, dropout, embedding_layer_q, embedding_layer_d):

        disc = Discriminator(samples_per_epoch, weight_decay, learning_rate, dropout, embedding_layer_q, embedding_layer_d)
        return disc
