from tensorflow.python.framework import graph_util
import tensorflow as tf
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def load_dictionary(path, title):
	with open(os.path.join(path, title+'.json')) as f:    
		return json.load(f)


if __name__ == '__main__':


	RNN_outputs_path = sys.argv[1]
	input_path = sys.argv[2]
	model_name = sys.argv[3]
	output_path = sys.argv[4]

	#RNN_outputs_path = '../build/28_frozen_models'
	#input_path = '../build/18_NN_dictionaries'
	#model_name = 'frozen_model_1'
	#output_path = '../build/18_NN_dictionaries'

	lemmas_dict_title = 'lemmas_dictionary'

	# Load tensorflow model
	with tf.gfile.GFile(os.path.join(RNN_outputs_path,'frozen_model_'+model_name), "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None, producer_op_list=None)

	# Load word vector node
	X1 = graph.get_tensor_by_name('import/X1:0')
	keep_prob = graph.get_tensor_by_name('import/keep_prob1:0')
	WV = graph.get_tensor_by_name('import/WV:0')


	with tf.Session(graph=graph) as sess:

		# convert the hot vectors to dense vectors using the tensorflow frozen_model
		# this will make inference much faster since word vectors are the most expensive part of the model
		dense_lemmas_dict = {}
		lemmas_dict = load_dictionary(input_path, lemmas_dict_title)

		for lemma in lemmas_dict.keys():
			dense_vector = sess.run([WV], {X1:[lemmas_dict[lemma]], keep_prob:1.0})

			# Convert the vector to normal python types because json doesn't like numpy types
			dense_lemmas_dict[lemma] = [float(n) for n in dense_vector[0][0]]

		with open(os.path.join(output_path,'dense_lemmas_dictionary.json'), 'w') as f:
			json.dump(dense_lemmas_dict, f)
