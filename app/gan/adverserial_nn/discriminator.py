from keras.layers import Dense, Bidirectional, Embedding, GRU, Concatenate
from keras.layers.core import Dropout
from keras.models import Model, Input, load_model, model_from_json

from gan.optimizer.AdamW import AdamW

import parameters as params


class Discriminator:
    def __init__(self, samples_per_epoch=0, weight_decay=None, learning_rate=None, dropout=0.2, embedding_layer_q=None, embedding_layer_d=None, model=None, sess=None):
        self.weight_decay = weight_decay
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.embeddings_layer_q: Embedding = embedding_layer_q
        self.embeddings_layer_d: Embedding = embedding_layer_d
        self.adamw = AdamW(lr=self.learning_rate, batch_size=params.DISC_BATCH_SIZE,
                            samples_per_epoch=self.samples_per_epoch, epochs=params.DISC_TRAIN_EPOCHS)
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

        self.lstm_q_in = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=True, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(self.embedded_sequences_q)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        self.lstm_q_out = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=False, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(self.lstm_q_in)

        self.sequence_input_d = Input(shape=(params.MAX_SEQUENCE_LENGTH_DOCS,), dtype='int32', name='input_doc')
        self.embedded_sequences_d = self.embeddings_layer_d(self.sequence_input_d)

        self.lstm_d_in = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=True, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(self.embedded_sequences_d)
        # this LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        self.lstm_d_out = Bidirectional(GRU(params.DISC_HIDDEN_SIZE_LSTM, kernel_initializer='random_uniform', return_sequences=False, activation='elu', dropout=self.dropout, recurrent_dropout=self.dropout))(self.lstm_d_in)

        self.x = Concatenate()([self.lstm_q_out, self.lstm_d_out])
        self.x = Dropout(self.dropout)(self.x)

        self.x = Dense(params.DISC_HIDDEN_SIZE_DENSE, activation='elu', kernel_initializer='random_uniform')(self.x)
        self.prob = Dense(1, kernel_initializer='random_uniform', activation='sigmoid', name='prob')(self.x)

        self.model = Model(inputs=[self.sequence_input_q, self.sequence_input_d], outputs=[self.prob])
        self.model.summary()

        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.adamw,
                           metrics=['accuracy'])

    def train(self, train_data_queries, train_data_documents, train_data_labels):

        from keras import callbacks

        reduce_lr = callbacks.EarlyStopping(monitor='loss', patience=3, verbose=0, mode='auto',
                                            baseline=None)

        return self.model.fit([train_data_queries, train_data_documents],
                              train_data_labels,
                              batch_size=params.DISC_BATCH_SIZE,
                              epochs=params.DISC_TRAIN_EPOCHS,
                              callbacks=[reduce_lr])

    def get_prob(self, train_data_queries, train_data_documents):
        return self.model.predict([train_data_queries, train_data_documents], batch_size=params.DISC_BATCH_SIZE)

    def get_reward(self, train_data_queries, train_data_documents):
        return (self.model.predict([train_data_queries, train_data_documents], batch_size=params.DISC_BATCH_SIZE) - 0.5) * 2

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

        disc = Discriminator(model=loaded_model)
        disc.model.compile(loss='binary_crossentropy', optimizer=disc.adamw, metrics=['accuracy'])
        return disc

    def load_weights_for_model(self, filepath_weights):
        # load weights into new model
        self.model.load_weights(filepath_weights)
        print("Loaded model from disk")

        self.model.compile(loss='binary_crossentropy', optimizer=self.adamw, metrics=['accuracy'])
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

        disc = Discriminator(model=loaded_model)
        disc.model.compile(loss='binary_crossentropy', optimizer=disc.adamw, metrics=['accuracy'])
        return disc

    @staticmethod
    def create_model(samples_per_epoch, weight_decay, learning_rate, dropout, embedding_layer_q, embedding_layer_d, sess):

        disc = Discriminator(samples_per_epoch, weight_decay, learning_rate, dropout, embedding_layer_q, embedding_layer_d, sess=sess)
        return disc
