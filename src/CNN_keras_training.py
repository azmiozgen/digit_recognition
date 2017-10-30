import numpy as np
import random, os, time

## Prepare data
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

training_images = np.array(training_images)
training_labels = np.array(training_labels)


## Preparing network

from keras import backend as K
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_accuracy
from keras.regularizers import l1_l2, l1, l2
from sklearn.utils import shuffle


## Hyper-parameters
FC_SIZE1 = 128
#FC_SIZE2 = 128
LR = 1e-6
MOMENTUM = 0.9
DECAY = 0.99
L2_REG = 0.01
EPOCH = 20
BATCH_SIZE = 100
KERNEL_SIZE1 = 3     ## One side (square)
KERNEL_SIZE2 = 3
KERNEL_SIZE3 = 5
KERNEL_SIZE4 = 3
KERNEL_SIZE5 = 3
FEATURE_MAP1 = 32     ## First conv layer feature maps
FEATURE_MAP2 = 32    ## Second conv layer feature maps
FEATURE_MAP3 = 16    ## Third conv layer feature maps
FEATURE_MAP4 = 32
FEATURE_MAP5 = 32
DROPOUT1 = 0.0
DROPOUT2 = 0.0
VALIDATION = 0.0
RESTORE = True

MODEL_PATH = "../model/CNN_keras/CNN_keras_5-2.h5"


## Reshape images
if K.image_data_format() == 'channels_first':
	training_images = training_images.reshape(training_images.shape[0], 1, 28, 28)
	input_shape = (1, 28, 28)
else:
	training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)


## Shuffle training set
training_images, training_labels = shuffle(training_images, training_labels, random_state=np.random.choice(range(EPOCH)))

if RESTORE and os.path.isfile(MODEL_PATH):
	model = load_model(MODEL_PATH)
	print("Model restored.")
else:
	model = Sequential()
	model.add(Conv2D(FEATURE_MAP1, (KERNEL_SIZE1, KERNEL_SIZE1), activation='relu', input_shape=input_shape))
	model.add(Conv2D(FEATURE_MAP2, (KERNEL_SIZE2, KERNEL_SIZE2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(FEATURE_MAP3, (KERNEL_SIZE3, KERNEL_SIZE3), activation='relu'))
	model.add(Conv2D(FEATURE_MAP4, (KERNEL_SIZE4, KERNEL_SIZE4), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(FEATURE_MAP5, (KERNEL_SIZE5, KERNEL_SIZE5), activation='relu'))
	model.add(Dropout(DROPOUT1))

	model.add(Flatten())
	model.add(Dense(FC_SIZE1, activation='relu'))
	model.add(Dropout(DROPOUT2))
	model.add(Dense(10, activation='softmax'))

	sgd = SGD(lr=LR, momentum=MOMENTUM, decay=DECAY, nesterov=True)
	adam = Adam(lr=LR)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[categorical_accuracy])

## See model summary
print model.summary()

# ## Set new LR and decay etc.
K.set_value(model.optimizer.lr, LR)
K.set_value(model.optimizer.decay, DECAY)
print "LR:",       K.get_value(model.optimizer.lr)
print "DECAY:",    K.get_value(model.optimizer.decay)

## Train
model.fit(training_images, training_labels, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=VALIDATION, shuffle=True)
model.save(MODEL_PATH)
print("Model saved.")


