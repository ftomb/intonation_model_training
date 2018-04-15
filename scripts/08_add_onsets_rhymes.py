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

def calculate_nucleus_index(phs):
	for i, ph in enumerate(phs):
		if '0' in ph.text or '1' in ph.text or '2' in ph.text:
			return i
	# If no indication about the nucleus is present assume the whole syll to be a rhyme
	return 0

def add_onsets_rhymes(title, input_path, output_path): 

	# Load the textgrid
	tg = tgt.read_textgrid(os.path.join(input_path,title+'.TextGrid'))

	# Load name of all tiers
	tier_names = tg.get_tier_names()

	# Select a tier whose name contains 'syllables'
	sylls_tier_name = [name for name in tier_names if 'sylls' in name][0]
	sylls_tier = tg.get_tier_by_name(sylls_tier_name)

	# Select a tier whose name contains 'phones'
	phones_tier_name = [name for name in tier_names if 'phones' in name][0]
	phones_tier = tg.get_tier_by_name(phones_tier_name)

	# Start an empty tier for onset-rhymes
	onset_rhyme_tier = tgt.IntervalTier()
	onset_rhyme_tier_name = [name for name in tier_names if 'words' in name][0].replace('words', 'OR')
	onset_rhyme_tier.name = onset_rhyme_tier_name

	onset_rhyme_intervals = []


	for syll in sylls_tier._get_annotations():

		#print(syll)
		phs = phones_tier.get_annotations_between_timepoints(syll.start_time, syll.end_time)
		
		nucleus_index = calculate_nucleus_index(phs)

		# If the first phone contains a number then it means the whole syll has no onset, so we only add a rhyme
		if nucleus_index == 0:
			onset_rhyme_intervals.append(tgt.Interval(syll.start_time, syll.end_time, 'R'))

		# If the onset is present add onset and rhyme intervals
		else:
			onset_rhyme_intervals.append(tgt.Interval(syll.start_time, phs[nucleus_index-1].end_time, 'O'))

			onset_rhyme_intervals.append(tgt.Interval(phs[nucleus_index].start_time, syll.end_time, 'R'))

	# Add all the intervals to the onset rhyme tier
	onset_rhyme_tier.add_annotations(onset_rhyme_intervals)

	# Add the onset rhyme tier to the TextGrid
	tg.add_tier(onset_rhyme_tier)
	
	# Move syll tier after the onset_rhyme_tier
	tg.delete_tier(sylls_tier_name)
	tg.add_tier(sylls_tier)

	tgt.write_to_file(tg, os.path.join(output_path,title+'.TextGrid'), format='short')


if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	#input_path = '../build/08_textgrid_syllables'
	#output_path = '../build/09_textgrid_onsets_rhymes'

	titles = load_titles(input_path, '.TextGrid')

	p = Pool()
	p.starmap(add_onsets_rhymes, [(title, input_path, output_path) for title in titles])