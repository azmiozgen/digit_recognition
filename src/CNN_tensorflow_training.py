import numpy as np
import os, sys

## Hyper-parameters
RATIO = 1.0   ## What ratio of total lines used for training (the rest is for validation)
FC_SIZE1 = 512
FC_SIZE2 = 256
LR = 1e-5
L2_REG = 0.01
EPOCH = 100
BATCH_SIZE = 500
KERNEL_SIZE1 = 3     ## One side (square)
KERNEL_SIZE2 = 5     ## One side (square)
KERNEL_SIZE3 = 5     ## One side (square)
FEATURE_MAP1 = 64    ## First conv layer feature maps
FEATURE_MAP2 = 32    ## Second conv layer feature maps
FEATURE_MAP3 = 32    ## Third conv layer feature maps
ADAPTIVE_LR = False

filePath = os.path.abspath(sys.argv[0])
fileName = os.path.basename(sys.argv[0])
repoPath = filePath.rstrip(fileName).rstrip("/").rstrip("src")
with open(repoPath + "data/extended4_train.csv") as f:
	lines_all = [line for line in f]

lines = [np.array(line.split(","), dtype='float32') for line in lines_all[1:]]

training_images = []
training_labels = []
validation_images = []
validation_labels = []
for line in lines[:int(len(lines) * RATIO)]:
	one_hot = np.zeros((10), dtype='float32')
	one_hot[int(line[0])] = 1.0
	training_labels.append(one_hot)
	training_images.append(line[1:])
training_set = [training_images, training_labels]
for line in lines[int(len(lines) * RATIO):]:
	one_hot = np.zeros((10), dtype='float32')
	one_hot[int(line[0])] = 1.0
	validation_labels.append(one_hot)
	validation_images.append(line[1:])
validation_set = [validation_images, validation_labels]

## Constants
TRAINING_SIZE = int(len(lines) * RATIO)
VALIDATION_SIZE = len(lines) - TRAINING_SIZE
TRAINING_BATCH = TRAINING_SIZE / BATCH_SIZE
VALIDATION_BATCH = VALIDATION_SIZE / BATCH_SIZE

####################
## Create Batches ##
####################

training_batches = []   ## [[[batch1_images], [batch1_labels]], [[batch2_images], [batch2_labels]], ... ]
for i in range(0, TRAINING_SIZE, BATCH_SIZE):
	training_batch_image = training_set[0][i:i + BATCH_SIZE]
	training_batch_label = training_set[1][i:i + BATCH_SIZE]
	training_batches.append([training_batch_image, training_batch_label])

validation_batches = []
for i in range(0, VALIDATION_SIZE, BATCH_SIZE):
	validation_batch_image = validation_set[0][i:i + BATCH_SIZE]
	validation_batch_label = validation_set[1][i:i + BATCH_SIZE]
	validation_batches.append([validation_batch_image, validation_batch_label])

#######################
## Preparing network ##
#######################

import tensorflow as tf
import time
import math

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## Input, output vectors
x = tf.placeholder(tf.float32, [None, 784])  ## Here 'None' means that a dimension can be of any length
y_ = tf.placeholder(tf.float32, [None, 10])  ## Correct answers

## 1st conv layer
## [Kernel size1, Kernel size2, first layer channel, second layer channel channel]
with tf.variable_scope("layer1"):
	W_conv1 = tf.get_variable('W_conv1', shape=(KERNEL_SIZE1, KERNEL_SIZE1, 1, FEATURE_MAP1),
								initializer=tf.contrib.layers.xavier_initializer())
	b_conv1 = tf.get_variable('b_conv1', shape=(FEATURE_MAP1),
								initializer=tf.contrib.layers.xavier_initializer())

## [.., shape1, shape2, channel]
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## 2nd conv layer
with tf.variable_scope("layer2"):
	W_conv2 = tf.get_variable('W_conv2', shape=(KERNEL_SIZE2, KERNEL_SIZE2, FEATURE_MAP1, FEATURE_MAP2),
								initializer=tf.contrib.layers.xavier_initializer())
	b_conv2 = tf.get_variable('b_conv2', shape=(FEATURE_MAP2),
								initializer=tf.contrib.layers.xavier_initializer())

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## 3rd conv layer (no pooling)
with tf.variable_scope("layer3"):
	W_conv3 = tf.get_variable('W_conv3', shape=(KERNEL_SIZE3, KERNEL_SIZE3, FEATURE_MAP2, FEATURE_MAP3),
								initializer=tf.contrib.layers.xavier_initializer())
	b_conv3 = tf.get_variable('b_conv3', shape=(FEATURE_MAP3),
								initializer=tf.contrib.layers.xavier_initializer())

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

##
## Fully connected layers
##

## Image shape halves twice. (28x28) -> (14x14) -> (7x7) by max_pool_2x2
## Conv. layer does not change image size because of padding='SAME'
with tf.variable_scope("layer4"):
	W_fc1 = tf.get_variable('W_fc1', shape=(7 * 7 * FEATURE_MAP3, FC_SIZE1),
								initializer=tf.contrib.layers.xavier_initializer())
	b_fc1 = tf.get_variable('b_fc1', shape=(FC_SIZE1),
								initializer=tf.contrib.layers.xavier_initializer())

h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * FEATURE_MAP3])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

## Dropout1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.variable_scope("layer5"):
	W_fc2 = tf.get_variable('W_fc2', shape=(FC_SIZE1, FC_SIZE2),
								initializer=tf.contrib.layers.xavier_initializer())
	b_fc2 = tf.get_variable('b_fc2', shape=(FC_SIZE2),
								initializer=tf.contrib.layers.xavier_initializer())

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

## Dropout2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

## Softmax output
with tf.variable_scope("layer6"):
	W_fc3 = tf.get_variable('W_fc3', shape=(FC_SIZE2, 10),
								initializer=tf.contrib.layers.xavier_initializer())
	b_fc3 = tf.get_variable('b_fc3', shape=(10),
								initializer=tf.contrib.layers.xavier_initializer())

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

####################
## Cost, accuracy ##
####################

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv + 1e-10), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_) +
								L2_REG * tf.nn.l2_loss(W_conv1) +
								L2_REG * tf.nn.l2_loss(b_conv1) +
								L2_REG * tf.nn.l2_loss(W_conv2) +
								L2_REG * tf.nn.l2_loss(b_conv2))
# train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#############
## Session ##
#############

training_accuracy = 0.0
validation_accuracy = 0.0
last_validation_accuracy = validation_accuracy
t0 = time.time()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	##############
	## Training ##
	##############

	saver.restore(sess, repoPath + "model/CNN_tensorflow_extended4_submission2/CNN_tensorflow_extended4_submission2.ckpt")
	print("Model restored.")

	for i in xrange(EPOCH * TRAINING_BATCH):

		j = i % TRAINING_BATCH
		sess.run(train_step, feed_dict={x: training_batches[j][0], y_: training_batches[j][1], keep_prob: 0.75})
		batch_accuracy = sess.run(accuracy, feed_dict={x: training_batches[j][0], y_: training_batches[j][1], keep_prob: 1.0})
		training_accuracy += batch_accuracy

		if (i + 1) / float(TRAINING_BATCH) == (i + 1) / TRAINING_BATCH:
			print "Epoch",  (i + 1) / TRAINING_BATCH, "\n\tTraining accuracy: {0:f}".format(training_accuracy / TRAINING_BATCH)
			training_accuracy = 0.0
			np.random.shuffle(training_batches)

			################
			## Validating ##
			################

			if RATIO != 1.0:
				validation_accuracy = 0.0
				for k in xrange(VALIDATION_BATCH):
					batch_accuracy = sess.run(accuracy, feed_dict={x: validation_batches[k][0], y_: validation_batches[k][1], keep_prob: 1.0})
					validation_accuracy += batch_accuracy

				print "\tValidation accuracy: {0:f}".format(validation_accuracy / VALIDATION_BATCH)
				np.random.shuffle(validation_batches)

				############################
				## Adaptive learning rate ##
				############################

				if ADAPTIVE_LR:
					if validation_accuracy > last_validation_accuracy and LR >= 1e-4:
						LR += LR * 0.05       ## Increase learning rate by %5 in case of higher accuracy
					elif validation_accuracy > last_validation_accuracy  and LR < 1e-4:
						LR += LR * 0.90       ## Increase learning rate by %90 when learning is slower
					else:
						LR -= LR * 0.50       ## Half the learning rate in case of lower accuracy
					last_validation_accuracy = validation_accuracy
					print "\tLearning rate: ", LR

	print "Training time:", time.time() - t0

	save_path = saver.save(sess, repoPath + "model/CNN_tensorflow_extended4_submission2/CNN_tensorflow_extended4_submission2.ckpt")
	print("Model saved in file: %s" %save_path)
