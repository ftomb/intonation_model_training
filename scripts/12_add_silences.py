from multiprocessing import Pool
import subprocess
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

def add_silences(title, input_path, output_path):

	tg = tgt.read_textgrid(os.path.join(input_path,title+'.TextGrid'))

	# Load name of all tiers
	tier_names = tg.get_tier_names()

	# Select a tier whose name contains 'phones'
	phones_tier_name = [name for name in tier_names if 'phones' in name][0]
	phones_tier = tg.get_tier_by_name(phones_tier_name)

	# Replace all sil and sp intervals with <sil> tag 
	#store these intervals to a list so that we can add them to the other tiers
	sil_intervals = []
	for interval in phones_tier:
		if interval.text == 'sil' or interval.text == 'sp':
			interval.text = '<sil>'
			sil_intervals.append(interval)

	# WORDS TIER
	# Select a tier whose name contains 'words'
	words_tier_name = [name for name in tier_names if 'words' in name][0]
	words_tier = tg.get_tier_by_name(words_tier_name)

	# Add <sil> to words tier
	words_tier.add_annotations(sil_intervals)

	# LEMMAS TIER
	# Select a tier whose name contains 'lemmas'
	lemmas_tier_name = [name for name in tier_names if 'lemmas' in name][0]
	lemmas_tier = tg.get_tier_by_name(lemmas_tier_name)

	# Add <sil> to lemmas tier
	lemmas_tier.add_annotations(sil_intervals)

	# SYLLABLES TIER
	# Select a tier whose name contains 'sylls'
	sylls_tier_name = [name for name in tier_names if 'sylls' in name][0]
	sylls_tier = tg.get_tier_by_name(sylls_tier_name)

	# Add <sil> to syllables tier
	sylls_tier.add_annotations(sil_intervals)

	# POS TIER
	# Select a tier whose name contains 'pos'
	pos_tier_name = [name for name in tier_names if 'pos' in name][0]
	pos_tier = tg.get_tier_by_name(pos_tier_name)

	# Add <sil> to pos tier
	pos_tier.add_annotations(sil_intervals)

	# OR TIER
	# Select a tier whose name contains 'OR'
	onset_rhyme_name = [name for name in tier_names if 'OR' in name][0]
	onset_rhyme_tier = tg.get_tier_by_name(onset_rhyme_name)

	# Add <sil> to OR tier
	onset_rhyme_tier.add_annotations(sil_intervals)

	# BP TIER
	# Select a tier whose name contains 'bp'
	bp_tier_name = [name for name in tier_names if 'bp' in name][0]
	bp_tier = tg.get_tier_by_name(bp_tier_name)

	# Add <sil> to bp tier
	bp_tier.add_annotations(sil_intervals)

	# FP TIER
	# Select a tier whose name contains 'fp'
	fp_tier_name = [name for name in tier_names if 'fp' in name][0]
	fp_tier = tg.get_tier_by_name(fp_tier_name)

	# Add <sil> to fp tier
	fp_tier.add_annotations(sil_intervals)

	tgt.write_to_file(tg, os.path.join(output_path,title+'.TextGrid'), format='short')

if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	#input_path = '../build/12_textgrid_punctuation'
	#output_path = '../build/13_textgrid_silences'

	titles = load_titles(input_path, '.TextGrid')

	p = Pool()
	p.starmap(add_silences, [(title, input_path, output_path) for title in titles])