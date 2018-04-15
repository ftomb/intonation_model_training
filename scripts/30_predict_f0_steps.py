from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
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

def load_batch(input_path, title):
	with open(os.path.join(input_path,title+'.json')) as f:  
		return json.load(f)

def load_dictionary(path, title):
	with open(os.path.join(path, title+'.json')) as f:    
		return json.load(f)

def extract_latest_vector(v):
	a = np.array([v[-1]])
	return (a == a.max(axis=1)[:,None]).astype(int)

def convert_vectors_to_f0_steps(vectors, dictionary):
	return [int(key) for v in vectors for key, value in dictionary.items() if np.array_equal(value, v) == True]


if __name__ == '__main__':

	RNN_model_path = sys.argv[1]
	model_name = sys.argv[2]
	input_path = sys.argv[3]
	dictionaries_path = sys.argv[4]
	output_path = sys.argv[5]

	#RNN_model_path = '../build/34b_frozen_models'
	#input_path = '../build/36_inference_batches'
	#dictionaries_path = '../build/26_NN_dictionaries'
	#model_name = 
	#output_path = '../build/37_predicted_f0_steps'



	dictionary_title1 = 'sign_labels_dictionary'
	dictionary_title2 = 'magn_labels_dictionary'

	# Load tensorflow model
	with tf.gfile.GFile(os.path.join(RNN_model_path,"frozen_model_"+model_name), "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None, producer_op_list=None)

	# Load tensorflow variables
	WV = graph.get_tensor_by_name('import/WV:0')
	X2 = graph.get_tensor_by_name('import/X2:0')
	keep_prob1 = graph.get_tensor_by_name('import/keep_prob1:0')
	n_frames1 = graph.get_tensor_by_name('import/n_frames1:0')
	seq_len1 = graph.get_tensor_by_name('import/seq_len1:0')
	BiRNN = graph.get_tensor_by_name('import/BiRNN:0')
	cutoff = graph.get_tensor_by_name('import/cutoff:0')
	X4 = graph.get_tensor_by_name('import/X4:0')
	X5 = graph.get_tensor_by_name('import/X5:0')
	n_frames2 = graph.get_tensor_by_name('import/n_frames2:0')
	Y2_ = graph.get_tensor_by_name('import/Y2_:0')
	Y3_ = graph.get_tensor_by_name('import/Y3_:0')
	#print('Done loading Tensorflow model...')

	sign_dict = load_dictionary(dictionaries_path, dictionary_title1)
	magn_dict = load_dictionary(dictionaries_path, dictionary_title2)
	seed_step1 = [sign_dict['0']]
	seed_step2 = [magn_dict['0']]

	titles = load_titles(input_path, '.json')

	with tf.Session(graph=graph) as sess:
		for title in sorted(titles):
			print(title)

			# Load a batch
			length, x1, x2 = load_batch(input_path, title)

			# calculate the length of the whole batch
			train_seq_len = [length]

			# Initialize the f0 outputs with the seed value
			x4_recusive = seed_step1
			x5_recusive = seed_step2

			# Extract the output of the bidirectional layers
			BiRNN_output = sess.run([BiRNN], {WV:x1, X2:x2, n_frames1:length, seq_len1:train_seq_len, keep_prob1:1.0})
			BiRNN_output = BiRNN_output[0]

			# Loop over f0 outputs to compute one at a time
			for i in range(len(x2)-1):
				x4_next, x5_next = sess.run([Y2_, Y3_], {BiRNN:BiRNN_output, X4:x4_recusive, X5:x5_recusive, n_frames2:len(x4_recusive), cutoff:len(x4_recusive), keep_prob1:1.0})

				latest_x4 = extract_latest_vector(x4_next)
				latest_x5 = extract_latest_vector(x5_next)

				x4_recusive = np.concatenate((x4_recusive, latest_x4))
				x5_recusive = np.concatenate((x5_recusive, latest_x5))

			sign_labels = convert_vectors_to_f0_steps(x4_recusive, sign_dict)
			magn_labels = convert_vectors_to_f0_steps(x5_recusive, magn_dict)

			f0_labels = [sign_labels[i]*magn_labels[i] for i in range(len(sign_labels))]

			with open(os.path.join(output_path,title+'.json'), 'w') as f:
				json.dump(f0_labels, f)
