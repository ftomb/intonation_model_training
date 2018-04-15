from multiprocessing import Pool
from random import shuffle
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

def load_data(title, input_path):
	with open(os.path.join(input_path,title+'.json')) as f:   
		return json.load(f)

def generate_unary_batches(title, input_path, output_path):
	print(title)

	data = load_data(title, input_path)
	unary_batch = [len(data), data]

	with open(os.path.join(output_path,title+'.json'), 'w') as f:
		json.dump(unary_batch, f)


if __name__ == '__main__':


	input_path = sys.argv[1]
	input2_path = sys.argv[2]
	inference_output_path = sys.argv[3]

	#input_path = '../build/25_merged_labels'
	#input2_path = '../build/28_train_test_titles'
	#inference_output_path = '../build/35_inference_labels'

	with open(os.path.join(input2_path, 'augmented_split_titles.json')) as f:    
		titles = json.load(f)
	valid_titles = titles['valid']
	#titles = load_titles(input_path, '.json')

	titles = load_titles(input_path, '.json')
	titles = sorted(set(valid_titles).intersection(titles))

	# Generate inference batches
	p = Pool()
	p.starmap(generate_unary_batches, [(title, input_path, inference_output_path) for title in titles])