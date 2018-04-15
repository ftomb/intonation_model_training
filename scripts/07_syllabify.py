from syllabifier import syllabify, stringify
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

def add_syllables(title, input_path, syllabification_file_path, output_path):

	# Load language syllable structure for the syllabifier
	with open(syllabification_file_path) as f:   
		language_syllables =  json.load(f)


	# Load the textgrid
	tg = tgt.read_textgrid(os.path.join(input_path,title+'.TextGrid'))

	# Load name of all tiers
	tier_names = tg.get_tier_names()

	# Select a tier whose name contains 'words'
	words_tier_name = [name for name in tier_names if 'words' in name][0]
	words_tier = tg.get_tier_by_name(words_tier_name)

	# Select a tier whose name contains 'phones'
	phones_tier_name = [name for name in tier_names if 'phones' in name][0]
	phones_tier = tg.get_tier_by_name(phones_tier_name)

	# Start an empty tier for syllables
	syllable_tier = tgt.IntervalTier()
	syll_tier_name = [name for name in tier_names if 'words' in name][0].replace('words', 'sylls')
	syllable_tier.name = syll_tier_name

	# Syllabify one word at a time
	for w in words_tier._get_annotations():
		
		# For the current word, get all of its phones
		phs = phones_tier.get_annotations_between_timepoints(w.start_time, w.end_time)
		for ph in phs: 
			if ph.text == 'spn':
				ph.text = 'aa1'


		# Transform the string of phones into a string of syllables
		# Format: ph1 ph2 . ph3 ph4 ph5 . ph6 etc.
		s = stringify(syllabify(' '.join([ph.text for ph in phs]), language_syllables))

		# From string of syllables to a nested lists of phone indeces
		# Format: [[ph1_idx, ph2_idx, etc.], [ph3_idx, ph4_idx, etc.], etc.]

		sylls = [syll.split() for syll in s.split('.')]
		i = 0
		sylls_indeces = []
		for j, syll in enumerate(sylls):
			syll_indeces = []
			for k in range(0, len(syll)):
				syll_indeces.append(int(i))
				i += 1
			sylls_indeces.append(syll_indeces)

		# Extract the relevant intervals using the indeces
		sylls_intervals = [[phs[index] for index in ph_group] for ph_group in sylls_indeces]

		# Extract the stress for each syllable:
		# Format: [['0'], ['1'], etc.]
		sylls_stresses = [[char for char in ''.join(ph_group) if char.isdigit()==True] for ph_group in sylls]
		sylls_stresses = [ph_group if ph_group != [] else ['0'] for ph_group in sylls_stresses]

		#print(w)
		#print(sylls_indeces)
		#print(sylls_stresses)
		#print(sylls_intervals)

		syllable_intervals = [tgt.Interval(interval[0].start_time, interval[-1].end_time, str(sylls_stresses[i][0])) for i, interval in enumerate(sylls_intervals)]

		#print(syllable_intervals)
		syllable_tier.add_annotations(syllable_intervals)

	tg.add_tier(syllable_tier)

	tgt.write_to_file(tg, os.path.join(output_path,title+'.TextGrid'), format='short')

if __name__ == '__main__':

	input_path = sys.argv[1]
	syllabification_file_path = sys.argv[2]
	output_path = sys.argv[3]

	#input_path = '../build/04_textgrid'
	#syllabification_file_path = '../src/syllabification_files/OALD.json'
	#output_path = '../build/08_textgrid_syllables'

	titles = load_titles(input_path, '.TextGrid')

	p = Pool()
	p.starmap(add_syllables, [(title, input_path, syllabification_file_path, output_path) for title in titles])

