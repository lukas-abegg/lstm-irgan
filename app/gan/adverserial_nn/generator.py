from keras import backend as K
from keras.layers import Bidirectional, Embedding, GRU, Dense, Concatenate
from keras.layers.core import Dropout
from keras.models import Model, Input, load_model, model_from_json
from keras.optimizers import Adam, Adadelta

from gan.optimizer.AdamW import AdamW

import numpy as np
import parameters as params


class Generator:
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
        self.adam = Adam(lr=self.learning_rate)
        self.adadelta = Adadelta(lr=self.learning_rate)
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
        self.x = Dense(params.GEN_HIDDEN_SIZE_DENSE, activation='elu', kernel_initializer='random_uniform')(self.x)
        self.prob = Dense(2, kernel_initializer='random_uniform', activation='softmax', name='prob')(self.x)

        self.model = Model(inputs=[self.sequence_input_q, self.sequence_input_d, self.reward, self.important_sampling],
                           outputs=[self.prob])

        self.model.summary()

        self.model.compile(loss=self.loss(self.reward, self.important_sampling),
                           optimizer=self.adadelta,
                           metrics=['accuracy'])

    def loss(self, _reward, _important_sampling):
        def _loss(y_true, y_pred):
            y_true = K.print_tensor(y_true, message="y_true is: ")
            y_true = y_true
            y_pred = K.print_tensor(y_pred, message="y_pred is: ")
            log_action_prob = K.log(y_pred[:, 1]) + 1e-08
            log_action_prob = K.print_tensor(log_action_prob, message="log_action_prob is: ")
            loss = K.reshape(log_action_prob, [-1]) * K.reshape(_reward, [-1]) * K.reshape(_important_sampling, [-1])
            total_loss = - K.mean(loss)
            total_loss = K.print_tensor(total_loss, message="total_loss is: ")
            return total_loss

        return _loss

    def train(self, train_data_queries, train_data_documents, reward, important_sampling):
        labels = np.zeros((len(train_data_queries), 2))

        for q in train_data_queries:
            assert not np.any(np.isnan(q))

        for d in train_data_queries:
            assert not np.any(np.isnan(d))

        from keras import callbacks

        stop_nan = callbacks.TerminateOnNaN()

        return self.model.fit([train_data_queries, train_data_documents, reward, important_sampling],
                              labels,
                              epochs=params.GEN_TRAIN_EPOCHS,
                              batch_size=params.GEN_BATCH_SIZE,
                              callbacks=[stop_nan])

    def get_prob(self, train_data_queries, train_data_documents):
        input_reward = [0.0] * len(train_data_queries)
        input_reward = np.asarray(input_reward)
        input_important_sampling = [0.0] * len(train_data_queries)
        input_important_sampling = np.asarray(input_important_sampling)
        pred_scores = self.model.predict([train_data_queries, train_data_documents, input_reward, input_important_sampling],
                                         batch_size=params.GEN_BATCH_SIZE)

        # If you're training for cross entropy, you want to add a small number like 1e-8 to your output probability.
        scores = pred_scores[:, 1] + 1e-08
        return scores

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

        gen = Generator(model=loaded_model)
        gen.model.compile(loss=gen.loss(gen.reward, gen.important_sampling),
                          optimizer=gen.adamw, metrics=['accuracy'])
        return gen

    def load_weights_for_model(self, filepath_weights):
        # load weights into new model
        self.model.load_weights(filepath_weights)
        print("Loaded model from disk")

        self.model.compile(loss=self.loss(self.reward, self.important_sampling),
                          optimizer=self.adamw, metrics=['accuracy'])
        return self

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

        gen = Generator(model=loaded_model)
        gen.model.compile(loss=gen.loss(gen.reward, gen.important_sampling),
                          optimizer=gen.adamw, metrics=['accuracy'])
        return gen

    @staticmethod
    def create_model(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q,
                     embedding_layer_d, sess):

        gen = Generator(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q,
                        embedding_layer_d, sess=sess)
        return gen
