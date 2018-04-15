import numpy as np
import json
import tgt
import sys
import os


def load_titles(target_path, extension):
	titles = []
	for fn in os.listdir(target_path):
		basename, ext = os.path.splitext(fn)
		if ext == extension:
			titles.append(basename)
	return titles

def generate_f0_labels_dictionary(titles, input_path, output_path):

	f0_dict = {}
	sign_dict = {}
	magn_dict = {}
	
	# Tally up all f0 labels
	n_titles = len(titles)
	for i, title in enumerate(titles):
		#print(round(i/n_titles, 3))


		# Load f0 labels data
		with open(os.path.join(input_path,title+'.json')) as f:    
			f0_labels = json.load(f)
		
		for datum in f0_labels:
			if datum[0] not in f0_dict.keys():
				if datum[0] == '-55':
					print(title)
				f0_dict[datum[0]] = 1
			else:
				f0_dict[datum[0]] += 1

		for datum in f0_labels:
			if datum[2] not in sign_dict.keys():
				sign_dict[datum[2]] = 1
			else:
				sign_dict[datum[2]] += 1

		for datum in f0_labels:
			if datum[4] not in magn_dict.keys():
				magn_dict[datum[4]] = 1
			else:
				magn_dict[datum[4]] += 1



	# Turn dictionary keys into list
	f0_labels_list = list(f0_dict.keys())
	sign_labels_list = list(sign_dict.keys())
	magn_labels_list = list(magn_dict.keys())

	# Convert the list into hot vector dictionary
	f0_hv_dict = {}
	for j, v in enumerate(f0_labels_list):
		f0_hv_dict[v] = [int(i) for i in np.eye(len(f0_labels_list), dtype=int)[j]]

	with open(os.path.join(output_path,'f0_labels_dictionary.json'), 'w') as f:
		json.dump(f0_hv_dict, f)

	# Convert the list into hot vector dictionary
	sign_hv_dict = {}
	for j, v in enumerate(sign_labels_list):
		sign_hv_dict[v] = [int(i) for i in np.eye(len(sign_labels_list), dtype=int)[j]]

	with open(os.path.join(output_path,'sign_labels_dictionary.json'), 'w') as f:
		json.dump(sign_hv_dict, f)

	# Convert the list into hot vector dictionary
	magn_hv_dict = {}
	for j, v in enumerate(magn_labels_list):
		magn_hv_dict[v] = [int(i) for i in np.eye(len(magn_labels_list), dtype=int)[j]]

	with open(os.path.join(output_path,'magn_labels_dictionary.json'), 'w') as f:
		json.dump(magn_hv_dict, f)

if __name__ == '__main__':


	input_path = sys.argv[1]
	output_path = sys.argv[2]

	#input_path = '../build/18_f0_labels'
	#output_path = '../build/20_NN_dictionaries'

	titles = load_titles(input_path, '.json')

	generate_f0_labels_dictionary(titles, input_path, output_path)