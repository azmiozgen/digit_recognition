import matplotlib.pyplot as plt
import numpy as np
import random

## Hyper-parameters
RATIO = 0.8   ## What ratio of total lines used for training (the rest is for validation)
FC_SIZE = 256
LR = 1e-3
L2_REG = 0.01
EPOCH = 5
BATCH_SIZE = 100
KERNEL_SIZE = 5     ## One side (square)
FEATURE_MAP1 = 64   ## First conv layer feature maps
FEATURE_MAP2 = 32   ## Second conv layer feature maps

with open("../data/train.csv") as f:
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

## Constants
TRAINING_SIZE = int(len(lines) * RATIO)
VALIDATION_SIZE = len(lines) - TRAINING_SIZE
TRAINING_BATCH = TRAINING_SIZE / BATCH_SIZE
VALIDATION_BATCH = VALIDATION_SIZE / BATCH_SIZE

## Preparing network

import tensorflow as tf
import time
import math

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1) * math.sqrt(2)
	# initial = tf.random_normal(shape, stddev=0.1) * math.sqrt(2)
	return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## Input, output vectors
x = tf.placeholder(tf.float32, [None, 784])  ## Here 'None' means that a dimension can be of any length
y_ = tf.placeholder(tf.float32, [None, 10])  ## Correct answers

## 1st conv layer
## [Kernel size1, Kernel size2, first layer channel, second layer channel channel]
W_conv1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, 1, FEATURE_MAP1])
b_conv1 = bias_variable([FEATURE_MAP1])

## [.., shape1, shape2, channel]
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

## 2nd conv layer
W_conv2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, FEATURE_MAP1, FEATURE_MAP2])
b_conv2 = bias_variable([FEATURE_MAP2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

## Fully connected layer

## Image shape halves twice. (28x28) -> (14x14) -> (7x7) by max_pool_2x2
## Conv. layer does not change image size because of padding='SAME'
W_fc1 = weight_variable([7 * 7 * FEATURE_MAP2, FC_SIZE])
b_fc1 = bias_variable([FC_SIZE])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * FEATURE_MAP2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## Softmax output
W_fc2 = weight_variable([FC_SIZE, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_) +
								L2_REG * tf.nn.l2_loss(W_conv1) +
								L2_REG * tf.nn.l2_loss(b_conv1) +
								L2_REG * tf.nn.l2_loss(W_conv2) +
								L2_REG * tf.nn.l2_loss(b_conv2))
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
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

# saver.restore(sess, "../model/CNN_tensorflow_submission/CNN_tensorflow_submission.ckpt")
# print("Model restored.")

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
		sess.run(train_step, feed_dict={x: training_batch_images[j], y_: training_batch_labels[j], keep_prob: 0.5})
		batch_accuracy = sess.run(accuracy, feed_dict={x: training_batch_images[j], y_: training_batch_labels[j], keep_prob: 1.0})
		total_training_accuracy += batch_accuracy

        # classification = sess.run(y_conv, feed_dict={x: training_batch_images[j], keep_prob: 1.0})
        # print classification
        # print training_batch_labels[j]

	################
	## Validating ##
	################


	print "Epoch",  (i + 1), "\n\tTraining accuracy: {0:f}".format(total_training_accuracy / TRAINING_BATCH)

	if RATIO != 1.0:
		total_validation_accuracy = 0.0
		for j in xrange(VALIDATION_BATCH):
			batch_accuracy = sess.run(accuracy, feed_dict={x: validation_batch_images[j], y_: validation_batch_labels[j], keep_prob: 1.0})
			total_validation_accuracy += batch_accuracy

		print "\tValidation accuracy: {0:f}".format(total_validation_accuracy / VALIDATION_BATCH)

print "Total time:", time.time() - t0

save_path = saver.save(sess, "../model/CNN_tensorflow_submission/CNN_tensorflow_submission.ckpt")
print("Model saved in file: %s" %save_path)
sess.close()
