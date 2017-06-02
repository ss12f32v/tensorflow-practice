import tensorflow as tf 
import numpy as np
import pandas as pd 
 
data = pd.read_csv('heart_pre.csv')
print (data.iloc[1:3,1:10])
# print (data.shape[0])

learning_rate = 0.01
batch_size = 10
n_epochs = 1000


X = tf.placeholder(tf.float32, [batch_size, 9],name="place_holder1")
Y = tf.placeholder(tf.float32, [batch_size, 2],name="place_holder2")

# randomize the weights , initialize bias
w = tf.Variable(tf.random_normal(shape=[9,2],stddev=0.1),name="weights")
b = tf.Variable(tf.zeros([1,2]),name = "bias")
logits = tf.matmul(X, w) + b

entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels = Y)
loss = tf.reduce_mean(entropy)

# define training op
optimizer = tf.train.GradientDescentOptimizer(learning_rate =learning_rate).minimize(loss)

init = tf.global_variables_initializer()

# for i in range(data.shape[0]):
# 	if data.iloc[i,10] == 1:
# 		data.iloc[i,9]=0
# 	else:
# 		data.iloc[i,9]=1
# 	if data.iloc[i,4] == "Present":
# 		data.iloc[i,4]=1
# 	else:
# 		data.iloc[i,4]=0	
# data.to_csv("heart_pre.csv",sep='\t')

with tf.Session() as sess:

	sess.run(init)
	n_batches = int (data.shape[0]/ batch_size)

	for i in range (n_epochs):
		for j in range (n_batches):
			X_batch = data.iloc[j*batch_size:(j+1)*batch_size, 1:10]
			Y_batch = data.iloc[j*batch_size:(j+1)*batch_size, 10:12]
			sess.run(optimizer, feed_dict={X: X_batch, Y:Y_batch})

	#test model		
	correct_prediction = 0
	for j in range(n_batches):
		X_batch = data.iloc[j*batch_size:(j+1)*batch_size, 1:10].values
		Y_batch = data.iloc[j*batch_size:(j+1)*batch_size, 10:12].values
		
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits],feed_dict={X: X_batch, Y:Y_batch})
		preds = tf.nn.sigmoid(logits_batch)
		#print (tf.argmax([Y_batch,1]))
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
		correct_prediction += sess.run(accuracy)
	print ("Accuracy: {0}".format(correct_prediction/data.shape[0]))