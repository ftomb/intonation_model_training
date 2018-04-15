from multiprocessing import Pool
import subprocess
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

def load_titles(target_path, extension):
	titles = []
	for fn in os.listdir(target_path):
		basename, ext = os.path.splitext(fn)
		if ext == extension:
			titles.append(basename)
	return titles

def extract_linguistic_labels(title, input1_path, input2_path, output_path):

	# Load the f0_timepoints
	with open(os.path.join(input1_path,title+'.json')) as f:    
		f0_timepoints = json.load(f)

	# Load textgrid
	tg = tgt.read_textgrid(os.path.join(input2_path,title+'.TextGrid'))
	tier_names = tg.get_tier_names()

	pos_tier_name = [name for name in tier_names if 'pos' in name][0]

	# Extract features from TextGrid, except for phones
	labels = []
	for t in f0_timepoints:
		labels_sublist = []
		for tier in tier_names:
			if 'phones' not in tier and 'words' not in tier:
				label = tg.get_tier_by_name(tier).get_annotations_by_time(t)[0].text
				labels_sublist.append(label)
		labels.append(labels_sublist)

	# Extract syllable boundaries
	syllables_tier_name = [name for name in tier_names if 'sylls' in name][0]
	syllables_tier = tg.get_tier_by_name(syllables_tier_name)

	syllable_boundary_times = [syllables_tier[0].start_time]+[interval.end_time for interval in syllables_tier]
	syllable_boudaries = [['1'] if t in syllable_boundary_times else ['0'] for t in f0_timepoints]


	# Extract word boundaries
	wores_tier_name = [name for name in tier_names if 'words' in name][0]
	words_tier = tg.get_tier_by_name(wores_tier_name)

	word_boundary_times = [words_tier[0].start_time]+[interval.end_time for interval in words_tier]
	word_boudaries = [['1'] if t in word_boundary_times else ['0'] for t in f0_timepoints]

	syllable_word_boundaries = [syllable_boudaries[i]+word_boudaries[i] for i in range(len(syllable_boudaries))]

	labels = [syllable_word_boundaries[i]+labels[i] for i in range(len(labels))]

	with open(os.path.join(output_path,title+'.json'), 'w') as f:
		json.dump(labels, f)


if __name__ == '__main__':

	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	output_path = sys.argv[3]

	#input1_path = '../build/16_f0_timepoints'
	#input2_path = '../build/14_augmented_texgrid'
	#output_path = '../build/17_linguistic_labels'

	titles1 = load_titles(input1_path, '.json')
	titles2 = load_titles(input2_path, '.TextGrid')

	titles = list(set(titles1).intersection(titles2))
	
	p = Pool()
	p.starmap(extract_linguistic_labels, [(title, input1_path, input2_path, output_path) for title in titles])