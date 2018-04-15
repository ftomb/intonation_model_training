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

def load_wordlist(input_path):
	with open(os.path.join(input_path)) as f:
		return [l.strip().lower() for l in f]

def generate_lemmas_dictionary(titles, input1_path, input2_path, output_path):

	# Load a control wordlist of the most frequent words in the eng language
	wl = load_wordlist(input2_path)

	# Collect all of the lemmas from the textgrids in only keep the ones in the control wordlist
	# We use the list because we don't want to learn frequent but biased words such as character names or words that are genre-related
	lemmas_dict = {}

	for title in titles:

		# Load the textgrid
		tg = tgt.read_textgrid(os.path.join(input1_path,title+'.TextGrid'))

		# Load name of all tiers
		tier_names = tg.get_tier_names()

		# Select a tier whose name contains 'lemmas'
		lemmas_tier_name = [name for name in tier_names if 'lemmas' in name][0]
		lemmas_tier = tg.get_tier_by_name(lemmas_tier_name)

		# Tally up all the lemmas in the textgrids
		for interval in lemmas_tier:
			if interval.text not in lemmas_dict.keys():
				lemmas_dict[interval.text] = 1
			else:
				lemmas_dict[interval.text] += 1


	lemmas_tuples = sorted(lemmas_dict.items(), key=lambda x: x[1], reverse=True)

	# 0.8 means that of all the frequencies we keep the highest 80% (for stimuli this was 100%)
	# excluding the lowest frequency words might help make the model more robust
	lemmas_freqs = sorted(set([lemmas_tuple[1] for lemmas_tuple in lemmas_tuples]), reverse=True)
	lemmas_freqs = lemmas_freqs[:math.floor(len(lemmas_freqs)*0.8)]
	lemmas_list = [lemmas_tuple[0] for lemmas_tuple in lemmas_tuples if lemmas_tuple[1] in lemmas_freqs]


	lemmas_list = ['<unk>']+[w for w in wl if w in lemmas_list]

	# Convert the list into hot vector dictionary
	hv_dict = {}
	for j, v in enumerate(lemmas_list):
		hv_dict[v] = [int(i) for i in np.eye(len(lemmas_list), dtype=int)[j]]

	with open(os.path.join(output_path,'lemmas_dictionary.json'), 'w') as f:
		json.dump(hv_dict, f)


if __name__ == '__main__':

	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	input3_path = sys.argv[3]
	output_path = sys.argv[4]

	#input1_path = '../build/14_augmented_texgrid'
	#input2_path = '../build/21_wordlist'
	#output_path = '../build/20_NN_dictionaries'

	with open(os.path.join(input3_path, 'augmented_split_titles.json')) as f:    
		titles_json = json.load(f)
	train_titles = titles_json['train']

	textgrid_titles = load_titles(input1_path, '.TextGrid')
	titles = sorted(list(set(textgrid_titles).intersection(train_titles)))

	generate_lemmas_dictionary(titles, input1_path, input2_path, output_path)


