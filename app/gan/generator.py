import tensorflow as tf

from keras.layers import Input, Concatenate, B
from keras.models import Model
from keras.layers.core import Reshape, Dense, Activation, Lambda
from keras import backend as k
from keras import regularizers
from keras import optimizers

import app.parameters as params


class Generator:
    def __init__(self, feature_size, hidden_size, weight_decay, learning_rate, temperature=1.0):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.temperature = temperature
       
        self.pred_data = Input(shape=(None, self.feature_size))
        self.reward = Input(shape=(None,))
        self.important_sampling = Input(shape=(None,))

        self.Dense_1_query = Dense(self.hidden_size, input_dim=self.feature_size, activation='tanh',
                                    weights=layer_0_param, kernel_regularizer=regularizers.l2(self.weight_decay))(self.pred_data)
        self.Dense_2_query = Dense(1, input_dim=self.hidden_size,
                                    weights=layer_1_param, kernel_regularizer=regularizers.l2(self.weight_decay))(self.Dense_1_query)

        self.Dense_1_document = Dense(self.hidden_size, input_dim=self.feature_size, activation='tanh',
                                    weights=layer_0_param, kernel_regularizer=regularizers.l2(self.weight_decay))(self.pred_data)
        self.Dense_2_document = Dense(1, input_dim=self.hidden_size,
                                    weights=layer_1_param, kernel_regularizer=regularizers.l2(self.weight_decay))(self.Dense_1_document)

        # Given batch query-url pairs, calculate the matching score
        # For all urls of one query
        self.Dense_Input = Concatenate([self.Dense_2_query, self.Dense_2_document], axis=-1)
        self.score = Lambda(lambda x: x / self.temperature)(self.Dense_2_result)
        self.score = Reshape([-1])(self.score)
        self.prob = Activation('softmax')(self.score)
        
        self.model = Model(inputs=[self.pred_data, self.reward, self.important_sampling], outputs=[self.prob])
        
        self.model.summary()
        self.model.compile(loss=self.__loss(self.reward, self.important_sampling),
                           optimizer=optimizers.TFOptimizer(tf.train.GradientDescentOptimizer(self.learning_rate)),
                           metrics=['accuracy'])

    def __loss(self, _reward, _important_sampling):
        def _loss(y_true, y_pred):
            log_action_prob = k.log(y_pred)
            loss = - k.reshape(log_action_prob, [-1]) * k.reshape(_reward, [-1]) * k.reshape(_important_sampling, [-1])
            loss = k.mean(loss)
            return loss

        return _loss(_reward, _important_sampling)

    def get_score(self, pred_data):
        functor = k.function([self.model.layers[0].input] + [k.learning_phase()], [self.model.layers[4].output])
        layer_outs = functor([pred_data, 0.])
        return layer_outs

    def get_prob(self, documents):
        functor = k.function([self.model.layers[0].input] + [k.learning_phase()], [self.model.layers[5].output])
        layer_outs = functor([documents, 0.])
        return layer_outs

    def train(self, x, y):
        self.model.fit(x=x, y=y, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

    def save_model(self, filename):
        self.model.save_weights(filename)

    @staticmethod
    def create_model(feature_size):
        # call discriminator, generator
        gen = Generator(feature_size, params.DISC_HIDDEN_SIZE, params.DISC_WEIGHT_DECAY, params.DISC_LEARNING_RATE)
        return gen
