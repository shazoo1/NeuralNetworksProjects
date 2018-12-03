import tensorflow as tf, numpy as np

hello = tf.constant('herro wolrd!')

sess = tf.Session()
print (sess.run(hello))

a = tf.constant(23)
b = tf.constant(46)
print (sess.run(a + b))

n_samples, batch_size, num_steps = 1000,100,20000
X_data = np.random.uniform(1, 10, (n_samples,1))
y_data = 2 * X_data + 1 + np.random.normal(0,2,(n_samples,1))
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear_regression'):
    k = tf.Variable(tf.random_normal((1,1), stddev=0.0), name='slope')
    b = tf.Variable(tf.zeros((1,)), name='bias')

y_pred = tf.matmul(X, k) + b
loss = tf.reduce_mean((y - y_pred) ** 2)
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

display_step = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        indices =  np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        _, loss_val, k_val, b_val = sess.run([optimizer, loss, k, b],
                                             feed_dict={X : X_batch, y : y_batch})
        if (i+1) % display_step == 0:
            print('Epoch %d: %.8f, k=%.4f, b=%.4f' % (i+1, loss_val, k_val, b_val))
