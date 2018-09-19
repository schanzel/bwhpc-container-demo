from os.path import isfile

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers.recurrent import LSTM
from keras.layers import Dropout, Dense
from keras.models import load_model as load
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

import numpy as np


class BinaryClassifierLSTM:

    def __init__(self,
                 input_len: int,
                 num_features: int = 1,
                 num_layers: int = 2,
                 dropout: float = None,
                 load_model: bool = True,
                 filename: str = None):
        self.filename = filename

        if load_model and filename is not None and isfile(filename):
            self.model = load(filename)
            self.was_loaded = True
        else:
            self.model = self.build_network(num_layers, input_len, num_features, dropout)
            self.was_loaded = False

        self.model.summary()

    @staticmethod
    def build_network(num_layers: int,
                      input_len: int,
                      num_features: int,
                      dropout: float) -> Sequential:
        model = Sequential(name='binary_classifier_lstm')
        for i in range(num_layers):
            model.add(LSTM(100,
                           input_shape=(input_len, num_features),
                           return_sequences=(i < num_layers - 1))
                      )
            if dropout is not None:
                model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = RMSprop()
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    def save_model(self) -> None:
        self.model.save(self.filename)

def sample_gaussian_noise(n_samples=10000, seq_length=1000):
    return np.random.normal(0, 1, (n_samples, seq_length))

def sample_sin(n_samples=10000, x_vals=np.arange(0, 100, .1), max_offset=100, mul_range=[1, 2]):
    vectors = []
    for i in range(n_samples):
        offset = np.random.random() * max_offset
        mul = mul_range[0] + np.random.random() * (mul_range[1] - mul_range[0])
        sin = np.sin(offset + x_vals * mul) / 2 + .5
        vectors.append(sin)
    return np.array(vectors)


if __name__ == '__main__':
    gauss = sample_gaussian_noise(n_samples=10)
    zeros = np.zeros((len(gauss), 1))

    sin = sample_sin(n_samples=10)
    ones = np.ones((len(sin), 1))

    gauss = np.append(gauss, zeros, axis=1)
    sin = np.append(sin, ones, axis=1)

    data = np.append(gauss, sin, axis=0)

    x = data[:, :len(data[0])-1]
    y = data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, shuffle=True)

    shape_train = (x_train.shape[0], x_train.shape[1], 1)
    x_train = x_train.reshape(shape_train)

    shape_test = (x_test.shape[0], x_test.shape[1], 1)
    x_test = x_test.reshape(shape_test)

    net = BinaryClassifierLSTM(input_len=shape_train[1],
                               num_features=shape_train[2],
                               num_layers=2,
                               dropout=.2,
                               filename='binary_classifier_lstm.h5')

    hist = net.model.fit(x_train, y_train,
                         batch_size=1,
                         epochs=1,
                         verbose=True,
                         validation_data=(x_test, y_test))
    net.save_model()

