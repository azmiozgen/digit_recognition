import numpy as np
from keras import backend as K
from keras.models import load_model

MODEL_PATH = "../model/CNN_keras/CNN_keras_5-2.h5"

## Load model
model = load_model(MODEL_PATH)
print("Model restored.")

## Prepare test data
with open("../data/test.csv") as f:
	lines = [line for line in f]

test_images = [np.array(line.split(','), dtype='float32') for line in lines[1:]]
test_images = np.array(test_images)

## Reshape images
if K.image_data_format() == 'channels_first':
	test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
	input_shape = (1, 28, 28)
else:
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)


## Predict test data
predictions = model.predict(test_images, batch_size=100)
predictions = np.array([np.argmax(i) for i in predictions])

pred_set = zip(test_images, predictions)

## Write to .csv
with open("../data/CNN_keras_submission_5_2.csv", "w") as sub:
	sub.write("ImageId,Label\n")
	for i in xrange(len(predictions)):
		sub.write(str(i + 1) + "," + str(predictions[i]) + "\n")
	print("Submission file is created.")