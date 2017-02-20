import tensorflow as tf
import numpy as np
import os, sys
import time

filePath = os.path.abspath(sys.argv[0])
fileName = os.path.basename(sys.argv[0])
repoPath = filePath.rstrip(fileName).rstrip("/").rstrip("src")
with open(repoPath + "/data/test.csv") as f:
	lines = [line for line in f]

test_images = [np.array(line.split(','), dtype='float32') for line in lines[1:]]

## Constants
TEST_SIZE = len(test_images)
BATCH_SIZE = 200
BATCH = TEST_SIZE / BATCH_SIZE

batches = [test_images[k:k + BATCH_SIZE] for k in range(0, TEST_SIZE, BATCH_SIZE)]

## Hyper-parameters
FC_SIZE = 1024
KERNEL_SIZE = 5     ## One side (square)
FEATURE_MAP1 = 32  ## First conv layer feature maps
FEATURE_MAP2 = 64   ## Second conv layer feature maps

## Variables
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
with tf.variable_scope("layer1"):
	W_conv1 = tf.get_variable('W_conv1', shape=(KERNEL_SIZE, KERNEL_SIZE, 1, FEATURE_MAP1),
								initializer=tf.contrib.layers.xavier_initializer())
	b_conv1 = tf.get_variable('b_conv1', shape=(FEATURE_MAP1),
								initializer=tf.contrib.layers.xavier_initializer())
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
with tf.variable_scope("layer2"):
	W_conv2 = tf.get_variable('W_conv2', shape=(KERNEL_SIZE, KERNEL_SIZE, FEATURE_MAP1, FEATURE_MAP2),
								initializer=tf.contrib.layers.xavier_initializer())
	b_conv2 = tf.get_variable('b_conv1', shape=(FEATURE_MAP2),
								initializer=tf.contrib.layers.xavier_initializer())
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
with tf.variable_scope("layer3"):
	W_fc1 = tf.get_variable('W_fc1', shape=(7 * 7 * FEATURE_MAP2, FC_SIZE),
								initializer=tf.contrib.layers.xavier_initializer())
	b_fc1 = tf.get_variable('b_fc1', shape=(FC_SIZE),
								initializer=tf.contrib.layers.xavier_initializer())
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * FEATURE_MAP2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
with tf.variable_scope("layer4"):
	W_fc2 = tf.get_variable('W_fc1', shape=(FC_SIZE, 10),
								initializer=tf.contrib.layers.xavier_initializer())
	b_fc2 = tf.get_variable('b_fc2', shape=(10),
								initializer=tf.contrib.layers.xavier_initializer())
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#############
## Testing ##
#############

saver = tf.train.Saver()
t0 = time.time()
with tf.Session() as sess:
	saver.restore(sess, repoPath + "model/CNN_tensorflow_extended4_submission/CNN_tensorflow_extended4_submission.ckpt")
	print("Model restored.")

	with open(repoPath + "data/CNN_tensorflow_extended4_submission.csv", "w") as sub:
		sub.write("ImageId,Label\n")
		for i in xrange(BATCH):
			classification_batch = sess.run(tf.argmax(y_conv, 1), feed_dict={x: batches[i], keep_prob: 1.0})

			for j in xrange(len(classification_batch)):
				sub.write(str(j + 1 + i * BATCH_SIZE) + "," + str(classification_batch[j]) + "\n")

	print "Total testing time:", time.time() - t0
