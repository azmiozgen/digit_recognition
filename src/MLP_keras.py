## Prepare data
import numpy as np
import random, os, time

with open("../data/train.csv") as f:
	training_lines_all = [line for line in f]

training_lines = [np.array(line.split(","), dtype='float32') for line in training_lines_all[1:]]

training_images = []
training_labels = []
training_set = []
for line in training_lines:
	one_hot = np.zeros((10), dtype='float32')
	one_hot[int(line[0])] = 1.0
	training_labels.append(one_hot)
	training_images.append(np.array(line[1:], dtype='float32'))
	training_set.append((one_hot, line[1:]))

## Preparing network

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

## Hyper-parameters
HIDDEN_SIZE = 500
LR = 1e-3
BATCH_SIZE = 10
EPOCH = 10

MODEL_PATH = "../model/MLP_keras/MLP_keras_hidden{}".format(HIDDEN_SIZE)

model = Sequential()
model.add(Dense(HIDDEN_SIZE, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
sgd = SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

if os.path.isfile(MODEL_PATH):
	model.load_weights(MODEL_PATH)
	print("Model restored.")
model.fit(training_images, training_labels, epochs=EPOCH, batch_size=BATCH_SIZE)
model.save_weights(MODEL_PATH)

