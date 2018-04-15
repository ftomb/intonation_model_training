from multiprocessing import Pool
import subprocess
import nltk
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

def my_tagger(s):

	POS_tags = nltk.pos_tag(nltk.word_tokenize(s, language='english'))

	POS_tags = [tag for tag in POS_tags if tag[1] != '.' and tag[1] != ',' and tag[1] != ':' and tag[1] != "''" and tag[1] != "--" and tag[1] != "``" and tag[1] != ")" and tag[1] != "(" and tag[0] != "SYM"]

	for i, tag in enumerate(POS_tags):
		if tag[0] == "'s" or tag[0] == "'d" or tag[0] == "n't" or tag[0] == "'ve" or tag[0] == "'re" or tag[0] == "'ll":
			POS_tags[i-1:i+1] = [(POS_tags[i-1][0]+POS_tags[i][0], POS_tags[i-1][1]+POS_tags[i][1])]

	for i, tag in enumerate(POS_tags):
		if tag[0] == "'" and tag[1] == "POS":
			POS_tags[i-1:i+1] = [(POS_tags[i-1][0]+POS_tags[i][0], POS_tags[i-1][1]+POS_tags[i][1])]

	for i, tag in enumerate(POS_tags):
		if tag[0].lower() == "'t" and tag[1] == "''":
			POS_tags[i:i+2] = [(POS_tags[i][0]+POS_tags[i+1][0], POS_tags[i][1]+POS_tags[i+1][1])]

	POS_tags = [(tag[0].lower(), tag[1]) for tag in POS_tags if tag[1] != '.']

	return POS_tags

def add_pos(title, input1_path, input2_path, output_path):

	# Load the textgrid
	tg = tgt.read_textgrid(os.path.join(input1_path,title+'.TextGrid'))

	# Load name of all tiers
	tier_names = tg.get_tier_names()

	# Select a tier whose name contains 'words'
	words_tier_name = [name for name in tier_names if 'words' in name][0]
	words_tier = tg.get_tier_by_name(words_tier_name)

	# Start an empty tier for POS_tags
	pos_tier = tgt.IntervalTier()
	pos_tier_name = [name for name in tier_names if 'words' in name][0].replace('words', 'pos')
	pos_tier.name = pos_tier_name

	# Extract words intervals
	word_intervals = [w for w in words_tier._get_annotations()]

	# Extract words
	words = [w.text for w in words_tier._get_annotations()]

	# Load text
	txt = ''
	with open(os.path.join(input2_path,title+'.txt'), 'r', encoding='utf-8') as f:	
		for l in f:
			l = ' '.join(l.split())
			for char in l.replace('\n', ' ').replace('\t', ' '):
				txt += char

	
	# Try to use my own tagger from txt and see if it matches the words in the original word tier
	# If they don't match just use the list of words from the tier and feed them to the tagger (this option is less accurate)

	my_tags = my_tagger(txt)
	if len(my_tags) == len(words):

		# True for every mismatch between words in words_tier and words produced by my_tagger
		mismatches = [True for i, tag in enumerate(my_tags) if tag[0] != words[i]]
		
		# If everything matches up we can use my_tags, else we resort to the vanilla nltk one
		if True not in mismatches:
			POS_tags = my_tags

		else:
			POS_tags = nltk.pos_tag(words)

	else:
		print(title)
		POS_tags = nltk.pos_tag(words)

	pos_intervals = [tgt.Interval(interval.start_time, interval.end_time, POS_tags[i][1])  for i, interval in enumerate(word_intervals)]

	pos_tier.add_annotations(pos_intervals)

	tg.add_tier(pos_tier)

	tgt.write_to_file(tg, os.path.join(output_path,title+'.TextGrid'), format='short')


if __name__ == '__main__':

	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	output_path = sys.argv[3]

	#input1_path = '../build/09_textgrid_onsets_rhymes'
	#input2_path = '../build/05_txt'
	#output_path = '../build/10_textgrid_pos'

	titles1 = load_titles(input1_path, '.TextGrid')
	titles2 = load_titles(input2_path, '.txt')

	titles = list(set(titles1).intersection(titles2))

	p = Pool()
	p.starmap(add_pos, [(title, input1_path, input2_path, output_path) for title in titles])