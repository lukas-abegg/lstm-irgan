import os
import tensorflow as tf
from keras import initializers
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Activation
from keras import backend as k
from keras import regularizers
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Discriminator:
    def __init__(self, feature_size, hidden_size, weight_decay, learning_rate):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.model = Sequential()
        self.model.add(Dense(self.hidden_size, input_dim = self.feature_size, activation='tanh',
                             weights = layer_0_param, kernel_regularizer = regularizers.l2(self.weight_decay), kernel_initializer = initializers.random_normal(stddev=0.01)))
        self.model.add(Dense(1, input_dim = self.hidden_size,
                             weights = layer_1_param, kernel_regularizer = regularizers.l2(self.weight_decay)))
        self.model.add(Reshape([-1]))
        self.model.add(Activation('sigmoid'))
        self.model.summary()
        self.model.compile(loss = 'binary_crossentropy', optimizer = optimizers.TFOptimizer(tf.train.GradientDescentOptimizer(self.learning_rate)), metrics=['accuracy'])

    def train(self, pred_data, pred_data_label):
        self.model.train_on_batch(pred_data, pred_data_label)

    def get_preresult(self, pred_data):
        return (self.model.predict(pred_data) - 0.5 ) *2

    def get_reward(self, pred_data):
        functor = k.function([self.model.layers[0].input]+[k.learning_phase()], [self.model.layers[3].output])
        layer_outs  = functor ([ pred_data, 1.])
        return (layer_outs[0]  - 0.5) * 2

    def save_model(self, filename):
        self.model.save_weights(filename)
