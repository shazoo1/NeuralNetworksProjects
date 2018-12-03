import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../Data/Digits/', one_hot=True)


def relu():
    x = tf.placeholder(tf.float32, [None, 784])

    # hidden relu layer
    b_relu = tf.Variable(tf.truncated_normal(shape=[784], stddev=0.1))
    W_relu = tf.Variable(tf.truncated_normal(shape=[784, 784], stddev=0.1))

    hidden_relu_layer = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)

    #dropout
    keep_prob = tf.constant(0.5)
    hidden_relu_layer = tf.nn.dropout(hidden_relu_layer, keep_prob=keep_prob)

    # 2 layer
    b = tf.Variable(tf.zeros([10]))
    W = tf.Variable(tf.zeros([784, 10]))

    y = tf.nn.softmax(tf.matmul(hidden_relu_layer, W) + b)

    # init
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.global_variables_initializer()

    # start training
    sess = tf.Session()
    sess.run(init)
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # results
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

relu()

