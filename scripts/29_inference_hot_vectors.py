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

	length, data = load_batch(input_path, title)
	tensors = [convert_to_hot_vectors(data, dictionaries, dim) for dim in dims]

	# convert np types to regular types because json cannot serialize them
	tensors = list([[list([float(n) for n in vector]) for vector in tensor] for tensor in tensors])
	hv_batch = [length]+tensors

	with open(os.path.join(output_path, title+'.json'), 'w') as f:
		json.dump(hv_batch, f)



if __name__ == '__main__':

	dictionaries_path = sys.argv[1]
	inference_input_path = sys.argv[2]
	inference_output_path = sys.argv[3]

	#dictionaries_path = '../build/27_merged_dictionaries'
	#inference_input_path = '../build/35_inference_labels'
	#inference_output_path = '../build/36_inference_batches'


	dictionaries_title = 'inference_dictionaries'

	# Load dictionaries
	dictionaries = load_dictionaries(dictionaries_path, dictionaries_title)


	# Define which dimensions you want to keep for each tensor
	x1_dim = [5] # word_vectors
	x2_dim = [0, 1, 2, 3, 4, 6, 7] # other linguistic labels

	dims = [x1_dim, x2_dim]


	# Generate hot vector batches for inferencing
	inference_titles = load_titles(inference_input_path, '.json')
	inference_titles = inference_titles

	p = Pool()
	p.starmap(convert_batch, [(title, inference_input_path, inference_output_path, dictionaries, dims) for title in inference_titles])