from keras import initializers
from keras.layers.core import Reshape, Dropout
from keras.layers import Bidirectional, Embedding, LSTM
from keras import backend as K
from keras import regularizers
from keras.layers import Dense, Concatenate, Activation
from keras.models import Model, Input
from keras.models import save_model, load_model

import app.parameters as params


class Discriminator:
    def __init__(self, weight_decay=None, learning_rate=None, embedding_layer_q=None, embedding_layer_d=None, model=None):
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
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
        sequence_input_q = Input(shape=(params.MAX_SEQUENCE_LENGTH_QUERIES,), dtype='int32')
        embedded_sequences_q = self.embeddings_layer_q(sequence_input_q)
        lstm_q_1 = Bidirectional(LSTM(units=params.DISC_HIDDEN_SIZE_LSTM, input_dim=params.EMBEDDING_DIM))(embedded_sequences_q)
        lstm_q_2 = Bidirectional(LSTM(units=params.DISC_HIDDEN_SIZE_LSTM, input_dim=params.DISC_HIDDEN_SIZE_LSTM))(lstm_q_1)
        lstm_out_q = Dropout(0.2)(lstm_q_2)

        sequence_input_d = Input(shape=(params.MAX_SEQUENCE_LENGTH_DOCUMENTS,), dtype='int32')
        embedded_sequences_d = self.embeddings_layer_q(sequence_input_d)
        lstm_d_1 = Bidirectional(LSTM(units=params.DISC_HIDDEN_SIZE_LSTM, input_dim=params.EMBEDDING_DIM))(embedded_sequences_d)
        lstm_d_2 = Bidirectional(LSTM(units=params.DISC_HIDDEN_SIZE_LSTM, input_dim=params.DISC_HIDDEN_SIZE_LSTM))(lstm_d_1)
        lstm_out_d = Dropout(0.2)(lstm_d_2)

        x = Concatenate([lstm_out_q, lstm_out_d])

        x = Dense(units=params.DISC_HIDDEN_SIZE_DENSE, activation='tanh',
                  kernel_regularizer=regularizers.l2(self.weight_decay),
                  kernel_initializer=initializers.random_normal(stddev=0.01))(x)
        x = Dense(units=1,
                  kernel_regularizer=regularizers.l2(self.weight_decay))(x)

        score = Reshape([-1])(x)
        prob = Activation('sigmoid')(score)

        model = Model(inputs=[sequence_input_q, sequence_input_d], outputs=[prob])
        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

    def train(self, train_data_queries, train_data_documents, train_data_label):
        self.model.train_on_batch([train_data_queries, train_data_documents], train_data_label)

    def get_preresult(self, train_data_queries, train_data_documents):
        return (self.model.predict([train_data_queries, train_data_documents]) - 0.5) * 2

    def get_reward(self, train_data_queries, train_data_documents):
        inp = self.model.input
        functor = K.function([inp] + [K.learning_phase()], [self.model.layers[13].output])
        layer_outs = functor([[train_data_queries, train_data_documents], 1.])
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
    def create_model(embedding_layer_q, embedding_layer_d):

        disc = Discriminator(params.DISC_WEIGHT_DECAY, params.DISC_LEARNING_RATE,
                             embedding_layer_q, embedding_layer_d)
        return disc

