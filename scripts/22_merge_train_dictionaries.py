import json
import sys
import os


def load_dictionary(path, title):
	with open(os.path.join(path, title+'.json')) as f:    
		return json.load(f)

if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	#input_path = '../build/20_NN_dictionaries'
	#output_path = '../build/22_merged_dictionaries'

	# Load dictionaries
	onsets_rhymes_dict_title = 'onsets_rhymes'
	syllables_dict_title = 'syllables'
	syllable_boundaries_dict_title = 'syllable_boundaries'
	pos_dict_title = 'pos_dictionary'
	lemmas_dict_title = 'lemmas_dictionary'
	word_boundaries_dict_title = 'word_boundaries'
	punctuation_dict_title = 'punctuation_dictionary'
	f0_labels_dict_title = 'f0_labels_dictionary'
	sign_labels_dict_title = 'sign_labels_dictionary'
	magn_labels_dict_title = 'magn_labels_dictionary'


	# arrange dicionaries in the same order as the labels
	dictionary_titles = [syllable_boundaries_dict_title, 
							word_boundaries_dict_title, 
							onsets_rhymes_dict_title, 
							syllables_dict_title, 
							pos_dict_title, 
							lemmas_dict_title, 
							punctuation_dict_title, 
							punctuation_dict_title, 
							f0_labels_dict_title, 
							f0_labels_dict_title, 
							sign_labels_dict_title, 
							sign_labels_dict_title, 
							magn_labels_dict_title, 
							magn_labels_dict_title
							]

	dictionaries = [load_dictionary(input_path,title) for title in dictionary_titles]

	with open(os.path.join(output_path, 'training_dictionaries.json'), 'w') as f:
		json.dump(dictionaries, f)

