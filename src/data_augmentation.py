from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import os

# Load data
with open("../data/train.csv") as f:
    lines = [line for line in f]
lines = [np.array(line.split(","), dtype='float32') for line in lines[1:]]

training_images = []
training_labels = []
for line in lines:
    one_hot = np.zeros((10), dtype='float32')
    one_hot[int(line[0])] = 1.0
    training_labels.append(one_hot)
    training_images.append(line[1:])
training_images = np.array(training_images)
training_labels = np.array(training_labels)
training_set = [training_images, training_labels]
training_images_origin = training_images[:]

## Reshape
training_images = training_images.reshape(training_images.shape[0], 1, 28, 28)

# Convert from int to float
training_images = training_images.astype('float32')

# Define data preparation
datagen = ImageDataGenerator(featurewise_center=False,
							samplewise_center=False,
							featurewise_std_normalization=False,
							samplewise_std_normalization=False,
							zca_whitening=False,
							rotation_range=30.,
							width_shift_range=0.2,
							height_shift_range=0.2,
							shear_range=0.1,
							zoom_range=0.2,
							channel_shift_range=0.05,
							fill_mode='nearest',
							cval=0.,
							horizontal_flip=False,
							vertical_flip=False,
							rescale=None,
							dim_ordering=K.image_dim_ordering())

# Fit parameters from data
datagen.fit(training_images)

BATCH_SIZE = 100
counter = 1
# Configure batch size and retrieve one batch of images
with open("../data/extended4_train.csv", mode="a") as f:
	for training_images, training_labels in datagen.flow(training_images, training_labels, batch_size=BATCH_SIZE, shuffle=True):
		for i in xrange(BATCH_SIZE):
			f.write(str(training_labels[i].nonzero()[0][0]) + ",")
			digit = training_images[i].reshape(784)
			digit = np.rint(digit).astype(np.int32)
			for pixel in digit:
				f.write(str(pixel) + ",")
			f.seek(-1, os.SEEK_END)
			f.truncate()
			f.write("\n")
		print "Batch:", counter
		counter += 1
		if counter == (42000 / BATCH_SIZE) * 3 + 1:
			break
