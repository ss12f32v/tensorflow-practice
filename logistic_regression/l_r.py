import time 
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)	

learning_rate = 0.01
batch_size = 128
n_epochs = 25


# MNIST data is  of shape 28*28 = 784
# Each image is represented by a 1*784 tensor

X = tf.placeholder(tf.float32, [batch_size, 784],name="place_holder1")
Y = tf.placeholder(tf.float32, [batch_size, 10],name="place_holder2")

# randomize the weights , initialize bias
w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.1),name="weights")
b = tf.Variable(tf.zeros([1,10]),name = "bias")

logits = tf.matmul(X, w) + b

# define loss function ( cross entrophyy)
# the softmax layer will applied by the function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels = Y)
loss = tf.reduce_mean(entropy)

# define training op
# using basic gradient descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate =learning_rate).minimize(loss)

#  Remember to initialize all the variable!

init = tf.global_variables_initializer()


with tf.Session() as sess:
	# visualize the model
	writer = tf.summary.FileWriter('./graphs', sess.graph)

	sess.run(init)
	#Compute how many batches
	n_batches = int (MNIST.train.num_examples/batch_size) 
	
	for i in range(n_epochs):
		for _ in range(n_batches):
			X_batch, Y_batch = MNIST.train.next_batch(batch_size)
			#sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
			sess.run(optimizer, feed_dict={X: X_batch, Y:Y_batch})

	#w_value , b_value  = sess.run([w,b])
	#print(w_value, b_value)
	

# Now we test the model
	n_batches = int(MNIST.test.num_examples/batch_size)
	correct_prediction = 0
	for i in range(n_batches):
		X_batch, Y_batch = MNIST.test.next_batch(batch_size)
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits],feed_dict={X: X_batch, Y:Y_batch})
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
		correct_prediction += sess.run(accuracy)

	print ("Accuracy {0}".format(correct_prediction/MNIST.test.num_examples))
	#print ("Accuracy:"),(format(correct_prediction/MNIST.test.num_examples))
writer.close()