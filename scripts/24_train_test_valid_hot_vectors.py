from multiprocessing import Pool
import numpy as np
import subprocess
import math
import json
import sys
import os

def load_titles(target_path, extension):
	titles = []
	for fn in os.listdir(target_path):
		basename, ext = os.path.splitext(fn)
		if ext == extension:
			titles.append(basename)
	return titles

def load_dictionaries(path, title):
	with open(os.path.join(path, title+'.json')) as f:    
		return json.load(f)

def load_batch(path, title):
	with open(os.path.join(path, title+'.json')) as f:    
		return json.load(f)

def convert_to_hot_vectors(data, dictionaries, dimensions):
	return np.array([np.concatenate([dictionaries[i][label] if label in dictionaries[i].keys() else dictionaries[i]['<unk>']  for i, label in enumerate(datum) if i in dimensions]) for datum in data])

def convert_batch(title, input_path, output_path, dictionaries, dims):
	try:

		length, data = load_batch(input_path, title)
		tensors = [convert_to_hot_vectors(data, dictionaries, dim) for dim in dims]
		# convert np types to regular types because json cannot serialize them
		tensors = list([[list([float(n) for n in vector]) for vector in tensor] for tensor in tensors])
		hv_batch = [length]+tensors

		with open(os.path.join(output_path, title+'.json'), 'w') as f:
			json.dump(hv_batch, f)
	except:
		print(title)
		pass


if __name__ == '__main__':


	dictionaries_path = sys.argv[1]

	train_input_path = sys.argv[2]
	test_input_path = sys.argv[3]
	valid_input_path = sys.argv[4]

	train_output_path = sys.argv[5]
	test_output_path = sys.argv[6]
	valid_output_path = sys.argv[7]



	#dictionaries_path = '../build/20_merged_dictionaries'
	#dictionaries_title = 'training_dictionaries'

	# Path for train batches, each batch is titles that have the same number of time steps
	#train_input_path = '../build/21_train_labels'
	#train_output_path = '../build/24_train_batches'

	# Path for test batches, each batch is titles that have the same number of time steps
	#test_input_path = '../build/22_test_labels'
	#test_output_path = '../build/25_test_batches'

	# Path for test batches, each batch is titles that have the same number of time steps
	#valid_input_path = '../build/23_valid_labels'
	#valid_output_path = '../build/26_valid_batches'




	dictionaries_title = 'training_dictionaries'

	# Load dictionaries
	dictionaries = load_dictionaries(dictionaries_path, dictionaries_title)

	# Define which dimensions you want to keep for each tensor
	x1_dim = [5] # word_vectors
	x2_dim = [0, 1, 2, 3, 4, 6, 7] # other linguistic labels
	x3_dim = [8] # f0_labels
	y1_dim = [9] # f0_labels
	x4_dim = [10] # sign_labels
	y2_dim = [11] # sign_labels
	x5_dim = [12] # magn_labels
	y3_dim = [13] # magn_labels

	dims = [x1_dim, x2_dim, x4_dim, y2_dim, x5_dim, y3_dim]


	# Generate hot vector batches for training, testing and validating
	train_batch_titles = load_titles(train_input_path, '.json')
	test_batch_titles = load_titles(test_input_path, '.json')
	valid_batch_titles = load_titles(valid_input_path, '.json')


	p = Pool()
	p.starmap(convert_batch, [(title, train_input_path, train_output_path, dictionaries, dims) for title in train_batch_titles])
	p.starmap(convert_batch, [(title, test_input_path, test_output_path, dictionaries, dims) for title in test_batch_titles])
	p.starmap(convert_batch, [(title, valid_input_path, valid_output_path, dictionaries, dims) for title in valid_batch_titles])
