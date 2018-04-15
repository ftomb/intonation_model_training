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

def group_titles_into_batches(titles, input_path, size_limit):

	# Group file titles based on the files' length so that each batch will have only data with the same number of time steps
	lengths_dict = {}
	for title in titles:
		length = len(load_data(title, input_path))
		if length not in lengths_dict.keys():
			lengths_dict[length] = [title]
		else:
			lengths_dict[length].append(title)

	batches = []

	for length, length_titles in lengths_dict.items():

		# Calculate how many files of a certain length would fit inside our size_limit 
		# (we take the ceiling so that the slicing step is at least >= 1, i.e. in case we have only one item for a certain length)
		slicing_step = math.ceil(size_limit/length)

		# generate list of there indeces where we are gonna slice the list
		idxs = [i for i in range(0, len(length_titles), slicing_step)]

		# here we make sure we only add the final index only if is not already there by chance (i.e. when the size_limit is a multiple of the slicing_step)
		if len(length_titles) not in idxs:
			idxs = idxs+[len(length_titles)]

		# Prepare indeces for slicing
		idx_slices = [[idxs[i], idxs[i+1]] for i in range(0, len(idxs)-1)]

		# Slice the list 
		slices = [length_titles[s[0]:s[1]] for s in idx_slices]
		
		for s in slices: 
			batches.append([length, s])

	return batches


def batchify_labels(title_batches, input_path, output_path, size_limit):

	batch_counter = 0

	for length, title_batch in title_batches:
		label_batch = []
		for title in title_batch:
			label_batch += load_data(title, input_path)

		if len(label_batch) > size_limit:
			label_batch = label_batch[:size_limit]
			length = size_limit

		batchified_labels = [length, label_batch]

		with open(os.path.join(output_path,str(batch_counter)+'.json'), 'w') as f:
			json.dump(batchified_labels, f)

		batch_counter += 1




def unary_batching(title, input_path, output_path):

	data = load_data(title, input_path)
	length = len(data)
	batchified_labels = [length, data]
	
	with open(os.path.join(output_path,title+'.json'), 'w') as f:
		json.dump(batchified_labels, f)



if __name__ == '__main__':

	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	train_output_path = sys.argv[3]
	test_output_path = sys.argv[4]
	valid_output_path = sys.argv[5]

	#input1_path = '../build/17_merged_labels'
	#input2_path = '../build/03_split_test_train_valid'
	#train_output_path = '../build/23_train_labels'
	#test_output_path = '../build/24_test_labels'
	#valid_output_path = '../build/25_valid_labels'

	merged_labels_titles = load_titles(input1_path, '.json')

	# Load test titles from the split_titles file
	with open(os.path.join(input2_path, 'augmented_split_titles.json')) as f:   
		titles = json.load(f)

	train_titles = sorted(set(titles['train']).intersection(merged_labels_titles))
	test_titles = sorted(set(titles['test']).intersection(merged_labels_titles))
	valid_titles = sorted(set(titles['valid']).intersection(merged_labels_titles))

	print(train_titles)

	unary_batch = True

	if unary_batch == True:

		p = Pool()
		p.starmap(unary_batching, [(title, input1_path, train_output_path) for title in train_titles])
		p.starmap(unary_batching, [(title, input1_path, test_output_path) for title in test_titles])
		p.starmap(unary_batching, [(title, input1_path, valid_output_path) for title in valid_titles])

	else:

		# this option puts all the files with the same length in the same batch.
		# this makes training fasters, but it also produces lower quality models

		# This is a limit on how long/large a batch will be
		# A batch is n_time_steps*n_files, 
		# where all the files have the same number of time steps
		size_limit = 10000

		# Generate train batches 
		train_title_batches = group_titles_into_batches(train_titles, input1_path, size_limit)
		print(len(train_title_batches))
		batchify_labels(train_title_batches, input1_path, train_output_path, size_limit)

		# Generate test batches
		test_title_batches = group_titles_into_batches(test_titles, input1_path, size_limit)
		print(len(test_title_batches))
		batchify_labels(test_title_batches, input1_path, test_output_path, size_limit)

		# Generate valid batches
		valid_title_batches = group_titles_into_batches(valid_titles, input1_path, size_limit)
		print(len(valid_title_batches))
		batchify_labels(valid_title_batches, input1_path, valid_output_path, size_limit)






