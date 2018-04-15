import numpy as np
import json
import sys
import os

def write_dictionary(title, output_path, label_list):
	hv_dict = {}
	for j, v in enumerate(label_list):
		hv_dict[v] = [int(i) for i in np.eye(len(label_list), dtype=int)[j]]
	with open(os.path.join(output_path,title+'.json'), 'w') as f:
		json.dump(hv_dict, f)

if __name__ == '__main__':

	output_path = sys.argv[1]
	#output_path = '../build/20_NN_dictionaries'

	# Create dictionary for syllable boundaries 
	syll_bound = ['<unk>', '0', '1']
	write_dictionary('syllable_boundaries', output_path, syll_bound)

	# Create dictionary for syllable boundaries 
	word_bound = ['<unk>', '0', '1']
	write_dictionary('word_boundaries', output_path, word_bound)

	# Create dictionary for onsets_rhymes
	onsets_rhymes = ['<unk>', '<sil>', 'O', 'R']
	write_dictionary('onsets_rhymes', output_path, onsets_rhymes)

	# Create dictionary for syllable boundaries 
	syllables = ['<unk>', '<sil>', '0', '1', '2']
	write_dictionary('syllables', output_path, syllables)


