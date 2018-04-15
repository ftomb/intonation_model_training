import numpy as np
import math
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

def generate_punctuation_dictionary(titles, input_path, output_path):

	punctuation_dict = {}

	# Gather bp intervals first
	for title in titles:

		# Load the textgrid
		tg = tgt.read_textgrid(os.path.join(input_path,title+'.TextGrid'))

		# Load name of all tiers
		tier_names = tg.get_tier_names()

		# Select a tier whose name contains 'bp'
		bp_tier_name = [name for name in tier_names if 'bp' in name][0]
		bp_tier = tg.get_tier_by_name(bp_tier_name)

		# Tally up all the pos in the textgrids
		for interval in bp_tier:
			if interval.text not in punctuation_dict.keys():
				punctuation_dict[interval.text] = 1
			else:
				punctuation_dict[interval.text] += 1

		# Select a tier whose name contains 'fp'
		fp_tier_name = [name for name in tier_names if 'fp' in name][0]
		fp_tier = tg.get_tier_by_name(fp_tier_name)

		# Tally up all the pos in the textgrids
		for interval in fp_tier:
			if interval.text not in punctuation_dict.keys():
				punctuation_dict[interval.text] = 1
			else:
				punctuation_dict[interval.text] += 1



	punct_tuples = sorted(punctuation_dict.items(), key=lambda x: x[1], reverse=True)

	# 0.8 means that of all the frequencies we keep the highest 80% (for stimuli this was 100%)
	# excluding the lowest frequency words might help make the model more robust
	punct_freqs = sorted(set([punct_tuple[1] for punct_tuple in punct_tuples]), reverse=True)
	punct_freqs = punct_freqs[:math.floor(len(punct_freqs)*0.8)]
	punct_list = [punct_tuple[0] for punct_tuple in punct_tuples if punct_tuple[1] in punct_freqs]


	# Extract the pos found in the corpus add the unk tag
	punct_list = ['<unk>']+punct_list

	# Convert the list into hot vector dictionary
	hv_dict = {}
	for j, v in enumerate(punct_list):
		hv_dict[v] = [int(i) for i in np.eye(len(punct_list), dtype=int)[j]]

	with open(os.path.join(output_path,'punctuation_dictionary.json'), 'w') as f:
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

	generate_punctuation_dictionary(titles, input1_path, output_path)