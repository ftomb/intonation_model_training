import numpy as np
import json
import math
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

def generate_pos_dictionary(titles, input_path, output_path):

	pos_dict = {}

	for title in titles:

		# Load the textgrid
		tg = tgt.read_textgrid(os.path.join(input_path,title+'.TextGrid'))

		# Load name of all tiers
		tier_names = tg.get_tier_names()

		# Select a tier whose name contains 'pos'
		pos_tier_name = [name for name in tier_names if 'pos' in name][0]
		pos_tier = tg.get_tier_by_name(pos_tier_name)

		# Tally up all the pos in the textgrids
		for interval in pos_tier:
			if interval.text not in pos_dict.keys():
				pos_dict[interval.text] = 1
			else:
				pos_dict[interval.text] += 1



	pos_tuples = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)

	# 0.8 means that of all the frequencies we keep the highest 80% (for stimuli this was 100%)
	# excluding the lowest frequency words might help make the model more robust
	pos_freqs = sorted(set([pos_tuple[1] for pos_tuple in pos_tuples]), reverse=True)
	pos_freqs = pos_freqs[:math.floor(len(pos_freqs)*0.8)]
	pos_list = [pos_tuple[0] for pos_tuple in pos_tuples if pos_tuple[1] in pos_freqs]


	# Extract the pos found in the corpus add the unk tag
	pos_list = ['<unk>']+pos_list

	# Convert the list into hot vector dictionary
	hv_dict = {}
	for j, v in enumerate(pos_list):
		hv_dict[v] = [int(i) for i in np.eye(len(pos_list), dtype=int)[j]]

	with open(os.path.join(output_path,'pos_dictionary.json'), 'w') as f:
		json.dump(hv_dict, f)


if __name__ == '__main__':

	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	output_path = sys.argv[3]

	#input_path = '../build/14_augmented_texgrid'
	#output_path = '../build/20_NN_dictionaries'

	textgrid_titles = load_titles(input1_path, '.TextGrid')

	with open(os.path.join(input2_path, 'augmented_split_titles.json')) as f:    
		titles_json = json.load(f)
	train_titles = titles_json['train']

	titles = sorted(list(set(textgrid_titles).intersection(train_titles)))

	generate_pos_dictionary(titles, input1_path, output_path)