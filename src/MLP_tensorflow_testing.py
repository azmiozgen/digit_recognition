import tensorflow as tf
import numpy as np
import os, sys
import time

filePath = os.path.abspath(sys.argv[0])
fileName = os.path.basename(sys.argv[0])
repoPath = filePath.rstrip(fileName).rstrip("/").rstrip("src")
with open(repoPath + "/data/test.csv") as f:
    lines_all = [line for line in f]

test_images = [np.array(line.split(','), dtype='float32') for line in lines_all[1:]]

## Hyper-parameters
HIDDEN_SIZE = 1000
BATCH_SIZE = 100
## Constant
TEST_SIZE = len(test_images)
BATCH = len(test_images) / BATCH_SIZE

## Inputs, Outputs, Variables
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.truncated_normal([784, HIDDEN_SIZE], stddev=0.1))
b1 = tf.Variable(tf.zeros([HIDDEN_SIZE]))
W2 = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y2 = tf.matmul(y1, W2) + b2

saver = tf.train.Saver()

## Launch the model in a session
# sess = tf.Session()
# if tf.__version__ == '0.10.0':
# 	sess.run(tf.initialize_all_variables())   ## tensorflow 0.10
# else:
# 	sess.run(tf.global_variables_initializer())   ## tensorflow 0.12


## Testing
with tf.Session() as sess:
	saver.restore(sess, repoPath + "model/MLP_tensorflow/MLP_tensorflow.ckpt")
	print("Model restored.")

	t0 = time.time()
	batches = [test_images[k:k + BATCH_SIZE] for k in range(0, TEST_SIZE, BATCH_SIZE)]

	with open(repoPath + "data/submission.csv", "w") as sub:
		sub.write("ImageId,Label\n")
		for i in xrange(BATCH):
			classification_batch = sess.run(tf.argmax(y2, 1), feed_dict={x: batches[i]})

			for j in xrange(len(classification_batch)):
				sub.write(str(j + 1 + i * BATCH_SIZE) + "," + str(classification_batch[j]) + "\n")

	print "Total testing time:", time.time() - t0
