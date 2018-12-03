import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../Data/DigitsConvo/', one_hot=True)

x_1d_fmt = tf.placeholder(tf.float32, [None, 784])

# Input: [batch size; height; width; channels]
x_2d_fmt = tf.reshape(x_1d_fmt, [-1, 28, 28, 1])

# Convolutional weights tensor [height; width; input channels; output channels]
w_lr_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_lr_1 = tf.Variable(tf.constant(0.1, shape=[32]))
conv_lr_1 = tf.nn.conv2d(x_2d_fmt, w_lr_1, strides=[1, 1, 1, 1], padding='SAME') + b_lr_1
conv_lr_1 = tf.nn.relu(conv_lr_1)

pool_lr_1 = tf.layers.max_pooling2d(conv_lr_1, 2, 2)

w_lr_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_lr_2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv_lr_2 = tf.nn.conv2d(pool_lr_1, w_lr_2, strides=[1, 1, 1, 1], padding='SAME') + b_lr_2
conv_lr_2 = tf.nn.relu(conv_lr_2)

pool_lr_2 = tf.layers.max_pooling2d(conv_lr_2, 2, 2)

h_pool_lr_2_flat = tf.reshape(pool_lr_2, [-1, 7 * 7 * 64])

w_full_conn = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_full_conn = tf.Variable(tf.constant(0.1, shape=[1024]))

full_conn_lr = tf.nn.relu(tf.matmul(h_pool_lr_2_flat, w_full_conn) + b_full_conn)
full_conn_lr = tf.layers.dropout(full_conn_lr, 0.4)

w_out = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_out = tf.Variable(tf.constant(0.1, shape=[10]))

y = tf.nn.softmax(tf.matmul(full_conn_lr, w_out) + b_out)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(150):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x_1d_fmt: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    indices = np.random.choice(10000, size=1000)
    epoch_accuracy = sess.run(accuracy, feed_dict={x_1d_fmt: mnist.test.images[indices], y_: mnist.test.labels[indices], })
    print("Accuracy of %s epoch is %s" % (i, epoch_accuracy))

indices = np.random.choice(10000, size=5000)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
final_accuracy = sess.run(accuracy, feed_dict={x_1d_fmt: mnist.test.images[indices], y_: mnist.test.labels[indices], })
print("Total accuracy is %s" % final_accuracy)
