from tensorflow.python.framework import graph_util
from shutil import copyfile
from random import shuffle
import tensorflow as tf
import numpy as np
import math
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def load_titles(target_path, extension):
	titles = []
	for fn in os.listdir(target_path):
		basename, ext = os.path.splitext(fn)
		if ext == extension:
			titles.append(basename)
	return titles

def load_titles_n_samples(input_path):
	titles = load_titles(input_path, '.json')
	# calculate n of time steps for the whole training corpus and testing corpus
	# we use the index 1 because each batch has the following structure: [length, x1, x2, x3, y1]
	#n_samples = sum([len(load_batch(title, input_path)[1]) for title in titles])
	n_samples = None

	return titles, n_samples

def load_batch(title, input_path):
	with open(os.path.join(input_path,title+'.json')) as f:  
		return json.load(f)

def weight(d1, d2):
	init_range = tf.sqrt(12.0/(d1+d2))
	initializer = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
	return tf.Variable(initializer([d1, d2]))

def bias(d2):
	return tf.Variable(tf.zeros([d2]))

def rnn_cell(recurrent_nodes):
	return tf.nn.rnn_cell.GRUCell(num_units=recurrent_nodes, activation=tf.nn.elu)

def rnn_layer(X, cell):
	return tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)

def bidirectional_rnn_layer(X, fw_cell, bw_cell, seq_len):
	return tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=X, dtype=tf.float32, sequence_length=seq_len)

def combine(L, W, B):
	return tf.add(tf.matmul(L, W), B)

def activate(L):
	return tf.nn.elu(L)

def calculate_loss(prediction, target):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target), name='loss')
	
def nadam_optimizer(loss):
	optimizer = tf.contrib.opt.NadamOptimizer().minimize(loss)
	return optimizer

def momentum_optimizer2(loss):
	optimizer = tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=True).minimize(loss)
	return optimizer

def momentum_optimizer(loss):

	# regularize loss by using L1
	l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
	weights = tf.trainable_variables()
	regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
	regularized_loss = loss + regularization_penalty 

	# regularize loss by using L2
	#regularized_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])*0.001

	# use momentum optimizer
	optimizer = tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=True).minimize(regularized_loss)
	return optimizer


def RNN(X1_d, X2_d, X4_d, X5_d, hidden_nodes, Y2_d, Y3_d):

	# This vector is for word embeddings
	X1 = tf.placeholder(tf.float32, [None, X1_d], name='X1')
	keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')

	W11 = weight(X1_d, int(hidden_nodes*4)) #1024
	B11 = bias(int(hidden_nodes*4))
	L11 = activate(combine(X1, W11, B11))
	L11 = tf.nn.dropout(L11, keep_prob1)

	W12 = weight(int(hidden_nodes*4), int(hidden_nodes*2)) #512
	B12 = bias(int(hidden_nodes*2))
	L12 = activate(combine(L11, W12, B12))
	L12 = tf.nn.dropout(L12, keep_prob1)

	W14 = weight(int(hidden_nodes*2), int(hidden_nodes)) #256
	B14 = bias(int(hidden_nodes))
	L13 = activate(combine(L12, W14, B14))
	L13 = tf.nn.dropout(L13, keep_prob1)

	# We store the word vector so that we can later do inference more efficiently
	WV = tf.identity(L13, name='WV')

	# Concat word vectors with other linguistic inputs
	X2 = tf.placeholder(tf.float32, [None, X2_d], name='X2')
	WV_X2 = tf.concat([WV, X2], 1)

	# Pass all inputs thru one commom feed-forward layer
	W2 = weight(X2_d+int(hidden_nodes), hidden_nodes*2) #256
	B2 = bias(hidden_nodes*2)
	L2 = activate(combine(WV_X2, W2, B2))
	L2 = tf.nn.dropout(L2, keep_prob1)

	# reshape the tensor so that the 2nd dimension is the n_timesteps (n_frames1)
	n_frames1 = tf.placeholder(tf.int32, name='n_frames1')
	L2_reshaped = tf.reshape(L2, [-1, n_frames1, hidden_nodes*2])

	# seq_len1 is needed by the bidirectional_dynamic_rnn
	seq_len1 = tf.placeholder(tf.int32, [None], name='seq_len1')

	# BiRNN: combine lingustic inputs backwards and forwards
	with tf.variable_scope('gru_1', initializer=tf.orthogonal_initializer(gain=tf.sqrt(2.0))):
		with tf.variable_scope('fw_1'):
			cell_fw_1 = rnn_cell(hidden_nodes*2) #512
		with tf.variable_scope('bw_1'):
			cell_bw_1 = rnn_cell(hidden_nodes*2) #512
		outputs1, states1 = bidirectional_rnn_layer(L2_reshaped, cell_fw_1, cell_bw_1, seq_len1)

	# The BiRNN returns forward and backward outputs, so we concat them
	# Then we go back to batch-major, as opposed to time-major (go back to len(batch)*nodes)
	outputs1_concatenated = tf.concat(outputs1, 2)
	outputs1_reshaped = tf.reshape(outputs1_concatenated, [-1, hidden_nodes*4]) #1024

	# Store the BiRNN output because, during inference 
	#we don't want to recalculate this value for each recursive output
	BiRNN = tf.identity(outputs1_reshaped, name='BiRNN')

	# We need the cutoff during the output recursion (to get rid of future BiRNN steps)
	cutoff = tf.placeholder(tf.int32, name='cutoff')
	BiRNN_slice = tf.slice(BiRNN, [0, 0], [cutoff, hidden_nodes*4]) #1024

	# Do this if you don't want to split sign and magnitude
	# Load first f0 vector and concat it with BiRNN_slice
	#X3 = tf.placeholder(tf.float32, [None, X3_d], name='X3')
	#BiRNN_slice_X3 = tf.concat([BiRNN_slice, X3], 1)

	# Load sign and magn vectors
	X4 = tf.placeholder(tf.float32, [None, X4_d], name='X4')
	X5 = tf.placeholder(tf.float32, [None, X5_d], name='X5')
	BiRNN_slice_X4_X5 = tf.concat([BiRNN_slice, X4, X5], 1)

	# Forward layer
	W31 = weight(hidden_nodes*4+X4_d+X5_d, hidden_nodes*2) #512
	B31 = bias(hidden_nodes*2)
	L31 = activate(combine(BiRNN_slice_X4_X5, W31, B31))
	L31 = tf.nn.dropout(L31, keep_prob1)

	# here we need n_frames2 for the second RNN, i.e. the one that predicts the actual f0
	n_frames2 = tf.placeholder(tf.int32, name='n_frames2')
	L31_reshaped = tf.reshape(L31, [-1, n_frames2, hidden_nodes*2])

	# First task: predict sign
	with tf.variable_scope('gru_2', initializer=tf.orthogonal_initializer(gain=tf.sqrt(2.0))):
		with tf.variable_scope('fw_2'):
			cell_fw_2 = rnn_cell(hidden_nodes*2) #512
		outputs2, states2 = rnn_layer(L31_reshaped, cell_fw_2)
	outputs2_reshaped = tf.reshape(outputs2, [-1, hidden_nodes*2])

	# Output layer
	W41 = weight(hidden_nodes*2, Y2_d)
	B41 = bias(Y2_d)
	Y2_ = combine(outputs2_reshaped, W41, B41)
	Y2_ = tf.identity(Y2_, name='Y2_')
	Y2 = tf.placeholder(tf.float32, [None, Y2_d], name='Y2')

	loss1 = calculate_loss(Y2_, Y2)
	correct1 = tf.equal(tf.argmax(Y2_, 1), tf.argmax(Y2, 1))
	accuracy1 = tf.reduce_mean(tf.cast(correct1, tf.float32))

	emitted_sign = tf.identity(Y2_)
	emitted_sign = tf.stop_gradient(emitted_sign)

	BiRNN_slice_X4_X5_Y2 = tf.concat([BiRNN_slice_X4_X5, emitted_sign], 1)

	# Second task: predict magnitude
	# Forward layer
	W32 = weight(hidden_nodes*4+X4_d+X5_d+Y2_d, hidden_nodes*2) #512
	B32 = bias(hidden_nodes*2)
	L32 = activate(combine(BiRNN_slice_X4_X5_Y2, W32, B32))
	L32 = tf.nn.dropout(L32, keep_prob1)

	# here we need n_frames2 for the second RNN, i.e. the one that predicts the actual f0
	L32_reshaped = tf.reshape(L32, [-1, n_frames2, hidden_nodes*2])

	with tf.variable_scope('gru_3', initializer=tf.orthogonal_initializer(gain=tf.sqrt(2.0))):
		with tf.variable_scope('fw_3'):
			cell_fw_3 = rnn_cell(hidden_nodes*2) #512
		outputs3, states3 = rnn_layer(L32_reshaped, cell_fw_3)
	outputs3_reshaped = tf.reshape(outputs3, [-1, hidden_nodes*2])

	# Output layer
	W42 = weight(hidden_nodes*2, Y3_d)
	B42 = bias(Y3_d)
	Y3_ = combine(outputs3_reshaped, W42, B42)
	Y3_ = tf.identity(Y3_, name='Y3_')
	Y3 = tf.placeholder(tf.float32, [None, Y3_d], name='Y3')

	loss2 = calculate_loss(Y3_, Y3)
	correct2 = tf.equal(tf.argmax(Y3_, 1), tf.argmax(Y3, 1))
	accuracy2 = tf.reduce_mean(tf.cast(correct2, tf.float32))
	
	accuracy = tf.concat([[accuracy1], [accuracy2]], 0)
	loss = loss1+loss2
	optimizer =  momentum_optimizer(loss)
	#optimizer =  nadam_optimizer(loss)

	return X1, X2, X4, X5, n_frames1, seq_len1, cutoff, n_frames2, Y2_, Y2, Y3_, Y3, loss, accuracy, optimizer, keep_prob1


if __name__ == '__main__':

	train_batches_path = sys.argv[1]
	n_epochs = int(sys.argv[2])
	RNN_outputs_path = sys.argv[3]
	models_path = sys.argv[4]

	#train_batches_path = '../build/24_train_batches/'
	#RNN_outputs_path = '../build/27_RNN_outputs/'
	#models_path = '../build/28_frozen_models/'


	train_titles, n_train_samples = load_titles_n_samples(train_batches_path)
	#train_titles = train_titles[:1]
	print(train_titles)


	# Calculate size of each input vector (ingore the first var because its the length)
	X1_d, X2_d, X4_d, Y2_d, X5_d, Y3_d = [len(v[0]) for v in load_batch(train_titles[0], train_batches_path)[1:]]
	print('Dimensions:', X1_d, X2_d, X4_d, Y2_d, X5_d, Y3_d)

	hidden_nodes = 256
	print('n_hidden_nodes:', hidden_nodes)

	# Define the RNN's graph
	X1, X2, X4, X5, n_frames1, seq_len1, cutoff, n_frames2, Y2_, Y2, Y3_, Y3, loss, accuracy, optimizer, keep_prob1 = RNN(X1_d, X2_d, X4_d, X5_d, hidden_nodes, Y2_d, Y3_d)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		e = 0
		best_accuracy = 0
		while True:
			e += 1
			print('Epoch:', e)
			shuffle(train_titles)
			epoch_message = 'Epoch: ' + str(e)
			epoch_train_loss = 0
			for title in train_titles:
				length, x1, x2, x4, y2, x5, y3 = load_batch(title, train_batches_path)
				train_seq_len = np.ones(int(len(x1)/length))*length
				_, title_loss = sess.run([optimizer, loss], {X1:x1, X2:x2, X4:x4, X5:x5, n_frames1:length, seq_len1:train_seq_len, n_frames2:length, cutoff:len(x1), Y2:y2, Y3:y3, keep_prob1:0.5})
				epoch_train_loss += title_loss
				print(title, length, 'Current Loss:', epoch_train_loss)
			print('Epoch Loss:', epoch_train_loss)
			a = 'Loss: ' + str(epoch_train_loss)


			saver = tf.train.Saver()
			saver.save(sess, os.path.join(RNN_outputs_path, 'models'))

			print('Saving frozen graph...')

			graph = tf.get_default_graph()
			input_graph_def = graph.as_graph_def()

			output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, ['Y2_', 'Y3_']) 

			with tf.gfile.GFile(os.path.join(RNN_outputs_path,'frozen_model'+'_'+str(e)), "wb") as f:
				f.write(output_graph_def.SerializeToString())

			copyfile(os.path.join(RNN_outputs_path,'frozen_model'+'_'+str(e)), os.path.join(models_path,'frozen_model'+'_'+str(e)))

			if e >= n_epochs:
				break

			


