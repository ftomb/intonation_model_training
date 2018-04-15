from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from nltk.corpus import wordnet
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

def clip_apostrophes(word):
	if word[-2:] == "'s":
		 word[:-2]
	if word.startswith("'"):
		word = word[1:]
	if word.endswith("'"):
		word = word[:-1]
	return word

def lemmatize_word(word, treebank_tag):

	word = clip_apostrophes(word)
	
	if treebank_tag.startswith('J'):
		return WordNetLemmatizer().lemmatize(word, wordnet.ADJ)
	elif treebank_tag.startswith('V'):
		return WordNetLemmatizer().lemmatize(word, wordnet.VERB)
	elif treebank_tag.startswith('N'):
		return WordNetLemmatizer().lemmatize(word, wordnet.NOUN)
	elif treebank_tag.startswith('R'):
		return WordNetLemmatizer().lemmatize(word, wordnet.ADV)
	else:
		return word

def add_lemmas(title, input1_path, output_path):

	# Load textgrid
	tg = tgt.read_textgrid(os.path.join(input1_path,title+'.TextGrid'))
	tier_names = tg.get_tier_names()

	# Load pos tier
	pos_tier_name = [name for name in tier_names if 'pos' in name][0]
	pos_tier = tg.get_tier_by_name(pos_tier_name)

	# Load words tier
	words_tier_name = [name for name in tier_names if 'words' in name][0]
	words_tier = tg.get_tier_by_name(words_tier_name)

	# Start empty lemmas tier
	lemmas_tier = tgt.IntervalTier()
	lemmas_tier_name = [name for name in tier_names if 'words' in name][0].replace('words', 'lemmas')
	lemmas_tier.name = lemmas_tier_name

	# Generate lemma intervals
	lemmas_intervals = [tgt.Interval(w_interval.start_time, w_interval.end_time, lemmatize_word(w_interval.text, pos_tier[i].text)) for i, w_interval in enumerate(words_tier)]

	# Add lemmas to tier
	lemmas_tier.add_annotations(lemmas_intervals)
	tg.add_tier(lemmas_tier)

	tgt.write_to_file(tg, os.path.join(output_path,title+'.TextGrid'), format='short')

if __name__ == '__main__':

	input1_path = sys.argv[1]
	output_path = sys.argv[2]

	#input1_path = '../build/10_textgrid_pos'
	#output_path = '../build/11_textgrid_lemmas'

	titles = load_titles(input1_path, '.TextGrid')

	p = Pool()
	p.starmap(add_lemmas, [(title, input1_path, output_path) for title in titles])