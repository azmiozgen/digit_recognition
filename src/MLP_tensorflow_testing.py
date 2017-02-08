import tensorflow as tf
import time

with open("../data/test.csv") as f:
    test_lines_all = [line for line in f]

test_images = [np.array(line, dtype='float32') for line in test_lines_all]

## Hyper-parameters
BATCH_SIZE = 100
## Constant
TEST_SIZE = len(test_images) / BATCH_SIZE

## Inputs, Outputs
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

## Launch the model in a session
sess = tf.Session()
if tf.__version__ == '0.10.0':
	sess.run(tf.initialize_all_variables())   ## tensorflow 0.10
else:
	sess.run(tf.global_variables_initializer())   ## tensorflow 0.12

saver = tf.train.Saver()

## Testing

saver.restore(sess, "../model/MLP_tensorflow/MLP_tensorflow.ckpt")
print("Model restored.")

t0 = time.time()
batches = [test_images[k:k + BATCH_SIZE] for k in range(0, TEST_SIZE, BATCH_SIZE)]
for j in xrange(BATCH):
	classification = sess.run(tf.argmax(y, 1), feed_dict={x: batches[j]})

print "Total testing time:", time.time() - t0
sess.close()
