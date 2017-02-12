import numpy as np
import os, sys
import random

## Hyper-parameters
RATIO = 1.0   ## What ratio of total lines used for training (the rest is for validation)
HIDDEN_SIZE = 1000
LR = 0.5e-5
BATCH_SIZE = 100
EPOCH = 1000

filePath = os.path.abspath(sys.argv[0])
fileName = os.path.basename(sys.argv[0])
repoPath = filePath.rstrip(fileName).rstrip("/").rstrip("src")
with open(repoPath + "/data/train.csv") as f:
    lines_all = [line for line in f]

lines = [np.array(line.split(","), dtype='float32') for line in lines_all[1:]]
random.shuffle(lines)

training_images = []
training_labels = []
training_set = []
validation_images = []
validation_labels = []
validation_set = []
for line in lines[:int(len(lines) * RATIO)]:
    one_hot = np.zeros((10), dtype='float32')
    one_hot[int(line[0])] = 1.0
    training_labels.append(one_hot)
    training_images.append(np.array(line[1:], dtype='float32'))
    training_set.append((one_hot, line[1:]))
for line in lines[int(len(lines) * RATIO):]:
	one_hot = np.zeros((10), dtype='float32')
	one_hot[int(line[0])] = 1.0
	validation_labels.append(one_hot)
	validation_images.append(np.array(line[1:], dtype='float32'))
	validation_set.append((one_hot, line[1:]))

## Constant
TRAINING_SIZE = int(len(lines) * RATIO)
VALIDATION_SIZE = len(lines) - TRAINING_SIZE
TRAINING_BATCH = TRAINING_SIZE / BATCH_SIZE
VALIDATION_BATCH = VALIDATION_SIZE / BATCH_SIZE

## Preparing network

import tensorflow as tf
import time


## Input vector
x = tf.placeholder(tf.float32, [None, 784])  ## Here 'None' means that a dimension can be of any length

## Creating 3-layers nn (1-input(784), 1-hidden(100), 1-output(10))
W1 = tf.Variable(tf.truncated_normal([784, HIDDEN_SIZE], stddev=0.1))
b1 = tf.Variable(tf.zeros([HIDDEN_SIZE]))
W2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))

y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y2 = tf.matmul(y1, W2) + b2

y_ = tf.placeholder(tf.float32, [None, 10])  ## Correct answers

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y2, y_))
train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

## Check if our prediction matches the truth (gives list of booleans)
correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Launch the model in a session
sess = tf.Session()
if tf.__version__ == '0.10.0':
	sess.run(tf.initialize_all_variables())   ## tensorflow 0.10
else:
	sess.run(tf.global_variables_initializer())   ## tensorflow 0.12

saver = tf.train.Saver()

##############
## Training ##
##############

saver.restore(sess, repoPath + "model/MLP_tensorflow/MLP_tensorflow.ckpt")
print("Model restored.")

validation_batches = [validation_set[k:k + BATCH_SIZE] for k in range(0, VALIDATION_SIZE, BATCH_SIZE)]
validation_batch_images = np.array([np.array([digit[1] for digit in batch], dtype='float32')
								  for batch in validation_batches], dtype='float32')
validation_batch_labels = np.array([np.array([digit[0] for digit in batch], dtype='float32')
								  for batch in validation_batches], dtype='float32')

t0 = time.time()
for i in xrange(EPOCH):
	random.shuffle(training_set)
	training_batches = [training_set[k:k + BATCH_SIZE] for k in range(0, TRAINING_SIZE, BATCH_SIZE)]

	training_batch_images = np.array([np.array([digit[1] for digit in batch], dtype='float32')
											    for batch in training_batches], dtype='float32')
	training_batch_labels = np.array([np.array([digit[0] for digit in batch], dtype='float32')
											    for batch in training_batches], dtype='float32')

	total_training_accuracy = 0.0
	for j in xrange(TRAINING_BATCH):
		sess.run(train_step, feed_dict={x: training_batch_images[j], y_: training_batch_labels[j]})
		batch_accuracy = sess.run(accuracy, feed_dict={x: training_batch_images[j], y_: training_batch_labels[j]})
		total_training_accuracy += batch_accuracy

	################
	## Validating ##
	################


	print "Epoch",  (i + 1), "\n\tTraining accuracy: {0:f}".format(total_training_accuracy / TRAINING_BATCH)

	if RATIO != 1.0:
		total_validation_accuracy = 0.0
		for j in xrange(VALIDATION_BATCH):
			batch_accuracy = sess.run(accuracy, feed_dict={x: validation_batch_images[j], y_: validation_batch_labels[j]})
			total_validation_accuracy += batch_accuracy

		print "\tValidation accuracy: {0:f}".format(total_validation_accuracy / VALIDATION_BATCH)

print "Total time:", time.time() - t0

save_path = saver.save(sess, repoPath + "model/MLP_tensorflow/MLP_tensorflow.ckpt")
print("Model saved in file: %s" %save_path)
sess.close()
