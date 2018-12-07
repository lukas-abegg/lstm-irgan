from keras import backend as K
from keras.layers import Bidirectional, Embedding, GRU, Dense, Activation, Lambda, Concatenate
from keras.layers.core import Reshape, Dropout
from keras.models import Model, Input, load_model, model_from_json

from gan.optimizer.AdamW import AdamW

import numpy as np
import parameters as params


class Generator:
    def __init__(self, samples_per_epoch=0, weight_decay=None, learning_rate=None, temperature=1.0, dropout=0.2, embedding_layer_q=None, embedding_layer_d=None, model=None, sess=None):
        self.weight_decay = weight_decay
        self.samples_per_epoch = samples_per_epoch
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.dropout = dropout
        self.embeddings_layer_q: Embedding = embedding_layer_q
        self.embeddings_layer_d: Embedding = embedding_layer_d
        self.reward = Input(shape=(None,), name='input_reward')
        self.important_sampling = Input(shape=(None,), name='input_imp_sampling')
        self.model: Model = self.__get_model(model)
        self.sess = sess

    def __get_model(self, model):
        if model is None:
            return self.__init_model()
        else:
            return model

    def __init_model(self):
        # create model

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

        # 0.2 should be replaced by self.temperature
        score = Lambda(lambda z: z / 0.2, name='raw_score')(x)
        score = Reshape([-1], name='score')(score)
        prob = Activation('softmax', name='prob')(score)

        model = Model(inputs=[sequence_input_q, sequence_input_d, self.reward, self.important_sampling], outputs=[prob])
        model.summary()

        adamw = AdamW(batch_size=params.GEN_BATCH_SIZE, samples_per_epoch=self.samples_per_epoch, epochs=params.GEN_TRAIN_EPOCHS)

        model.compile(loss=self.loss(self.reward, self.important_sampling),
                      optimizer=adamw,
                      metrics=[self.loss_metrics(self.reward, self.important_sampling)])

        return model

    def loss(self, _reward, _important_sampling):
        def _loss(y_true, y_pred):
            log_action_prob = K.log(y_pred)
            loss = - K.reshape(log_action_prob, [-1]) * K.reshape(_reward, [-1]) * K.reshape(_important_sampling, [-1])
            loss = K.mean(loss)
            return loss

        return _loss

    def loss_metrics(self, _reward, _important_sampling):
        def _metrics(y_true, y_pred):
            log_action_prob = K.log(y_pred)
            loss = - K.reshape(log_action_prob, [-1]) * K.reshape(_reward, [-1]) * K.reshape(_important_sampling, [-1])
            loss = K.mean(loss)
            return loss

        return _metrics

    def train(self, train_data_queries, train_data_documents, reward, important_sampling):
        return self.model.train_on_batch([train_data_queries, train_data_documents, reward, important_sampling], np.zeros([train_data_queries.shape[0]]))

    def get_prob(self, train_data_queries, train_data_documents):
        pred_scores = []
        i = 1
        while i <= len(train_data_queries):
            batch_index = i - 1
            if i + params.GEN_BATCH_SIZE <= len(train_data_queries):
                input_queries = train_data_queries[batch_index:batch_index + params.GEN_BATCH_SIZE]
                input_documents = train_data_documents[batch_index:batch_index + params.GEN_BATCH_SIZE]
            else:
                input_queries = train_data_queries[batch_index:len(train_data_queries)]
                input_documents = train_data_documents[batch_index:len(train_data_queries)]

            i += params.GEN_BATCH_SIZE

            inputs = self.model.inputs + [K.learning_phase()]
            out = self.model.get_layer('prob').output
            functor = K.function(inputs, [out])
            layer_outs = functor([input_queries, input_documents, 0.])
            pred_scores.extend(layer_outs[0])

        pred_scores = np.asarray(pred_scores)
        return pred_scores

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

        gen = Generator(model=loaded_model)
        return gen

    @staticmethod
    def create_model(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q, embedding_layer_d, sess):

        gen = Generator(samples_per_epoch, weight_decay, learning_rate, temperature, dropout, embedding_layer_q, embedding_layer_d, sess=sess)
        return gen
