from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss



def word2vec(batch_gen):
	# step 1-1. Instead of using hot vector , input the index of the words
	with tf.name_scope("data"):
		center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
		target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE,1], name='target_words')

	# step 1-2. Define the weight ( the embedding matrix )
	# Word is represent of the each row of the Weight
	# randomlize the matrix
	with tf.name_scope("embed"):
		embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE],-1.0,1.0),name='embed_matrix')

	# step 1-3. Inference  define the forward path of the graph
	# The function we will use :tf.nn.embedding _lookup(params, ids, partition_strategy='mod', name=None,validate_indices= True, max_norm=None)
	with tf.name_scope("loss"):
		embed = tf.nn.embedding_lookup(embed_matrix, center_words,name='embed')

	# step 1-4. Define the loss function
	# We use the NCE , the negative sampling assume K*Q(w)=1
	# These two sampling is to maximize the probability to real target word 
	# and minimize the probability of the other k noise 
		
		nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0 / EMBED_SIZE ** 0.5),name='nce_weight')
		nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]),name='nce_bias')

		loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
											biases=nce_bias,
											labels=target_words,
											inputs=embed,
											num_sampled=NUM_SAMPLED,
											num_classes=VOCAB_SIZE), name='loss')
	# step 1-5. Define optimizer
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

	init = tf.global_variables_initializer()

	#Phase2 start 
	# step 2-1.
	with tf.Session() as sess:
		sess.run(init)
		average_loss = 0.0
		writer = tf.summary.FileWriter('./my_graph', sess.graph)
		for index in range(NUM_TRAIN_STEPS):
			batch = next(batch_gen)
			loss_batch, _ = sess.run([loss,optimizer], feed_dict={center_words: batch[0], target_words: batch[1]})
			average_loss += loss_batch
			if (index + 1) % 2000 == 0:
				print('Average loss at step {}: {:5.1f}'.format(index + 1, average_loss / (index + 1)))
		writer.close()


if __name__ == '__main__':
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)
    #word2vec(None)