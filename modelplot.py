from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from keras import Input, Model

os.environ["PATH"] += os.pathsep + 'D:\Software\graphviz-2.38\releasebin/'

from datetime import datetime
from packaging import version
from keras.models import Sequential
from keras.layers import Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dense, LSTM
from keras.layers import Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dense, concatenate
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.ops.gen_summary_ops import SummaryWriter

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

model = Sequential(name="feedforward-bow-input")
model.add(Dense(10, input_dim=31646, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())  # (批)规范化层
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
from keras.utils import plot_model
import pydot
plot_model(model,to_file='FFNNmodel.png',show_shapes=True,show_layer_names=False)

main_input = Input(shape=(150,), dtype='float64')
embedder = Embedding(31646 + 1, 150, input_length=150, trainable=False)
embed = embedder(main_input)
cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu',kernel_initializer='lecun_uniform')(embed)
cnn1 = MaxPooling1D(pool_size=48)(cnn1)
cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu',kernel_initializer='lecun_uniform')(embed)
cnn2 = MaxPooling1D(pool_size=47)(cnn2)
cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu',kernel_initializer='lecun_uniform')(embed)
cnn3 = MaxPooling1D(pool_size=46)(cnn3)
cnn4 = Conv1D(256, 6, padding='same', strides=1, activation='relu',kernel_initializer='lecun_uniform')(embed)
cnn4 = MaxPooling1D(pool_size=45)(cnn4)
cnn5 = Conv1D(256, 7, padding='same', strides=1, activation='relu',kernel_initializer='lecun_uniform')(embed)
cnn5 = MaxPooling1D(pool_size=44)(cnn5)
cnn = concatenate([cnn1, cnn2, cnn3, cnn4, cnn5], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
main_output = Dense(2, activation='softmax')(drop)
model = Model(inputs=main_input, outputs=main_output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
plot_model(model,to_file='CNNmodel.png',show_shapes=True,show_layer_names=False)