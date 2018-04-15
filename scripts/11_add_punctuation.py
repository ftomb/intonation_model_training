from multiprocessing import Pool
import subprocess
import string
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

def find_next(l, item):
	for i, v in enumerate(l):
		if v == item:
			return i

def clean_word(w, title, txt):

	lw = [char for char in w]
	punt_places = [1 if char in string.punctuation else 0 for char in lw]
	start_index = find_next(punt_places, 0)

	rev_lw = lw[::-1]
	punt_places = [1 if char in string.punctuation else 0 for char in rev_lw]

	end_index = -find_next(punt_places, 0)

	if end_index == 0:
		end_index = None

	clean_word = w[start_index:end_index]
	
	return clean_word


def detect_non_words(w):
	test = sum([0 if char in string.punctuation else 1 for char in w])
	if test == 0:
		return '<punct>'
	else:
		return w


def add_punctuation(title, textgrid_path, txt_path, output_path): 

	txt = ''
	with open(os.path.join(txt_path,title+'.txt'), 'r', encoding='utf-8') as f:	
		for l in f:
			l = ' '.join(l.split())
			for char in l.replace('\n', ' ').replace('\t', ' ').lower():
				txt += char

	word_non_words = [detect_non_words(w) for w in txt.split()]

	# Exclude non-words such as -' , - etc.
	txt_words = [w for w in word_non_words if w != '<punct>']

	# Strip words of punctuation before and after the first/last alphanum
	txt_words = [clean_word(w, title, txt) for w in txt_words]

	tg = tgt.read_textgrid(os.path.join(textgrid_path,title+'.TextGrid'))

	# Load name of all tiers
	tier_names = tg.get_tier_names()

	# Select a tier whose name contains 'words'
	word_tier_name = [name for name in tier_names if 'words' in name][0]
	word_tier = tg.get_tier_by_name(word_tier_name)
	word_list = [w.text for w in word_tier._get_annotations()]
	

	if len(word_list) == len(txt_words):

		w_indices = []
		w_indices.append(0)
		start = 0
		for lw in txt_words:
			idx = txt.find(lw, start, len(txt))
			start = idx+len(lw)
			w_indices.append(idx)
			w_indices.append(idx+len(lw))
		w_indices.append(len(txt))

		p_indices = [[w_indices[i], w_indices[i+1]] for i in range(0, len(w_indices)-1, 2)]
		punctuation = [txt[i[0]:i[1]].replace(' ', '') for i in p_indices]
		punctuation[0] = 'start' + punctuation[0]
		punctuation[-1] = punctuation[-1] + 'end'
		punctuation = [p if p != '' else '_' for p in punctuation]

		bp = punctuation[0:-1]
		fp = punctuation[1:]

		word_durations = []
		for w in word_tier._get_annotations():
			word_durations.append(w)

		# here we go thru this list ([[w_dur1, w_dur2, etc.], [w_dur1, w_dur2, etc.], etc]) and we keep the first and the last duration of every word
		bp_tier = tgt.IntervalTier()
		bp_tier.name = [name for name in tier_names if 'words' in name][0].replace('words', 'bp')
		bp_intervals = [tgt.Interval(word_durations[i].start_time, word_durations[i].end_time, bp[i]) for i in range(0, len(word_durations))]
		bp_tier.add_annotations(bp_intervals)
		tg.add_tier(bp_tier)

		fp_tier = tgt.IntervalTier()
		fp_tier.name = [name for name in tier_names if 'words' in name][0].replace('words', 'fp')
		fp_intervals = [tgt.Interval(word_durations[i].start_time, word_durations[i].end_time, fp[i]) for i in range(0, len(word_durations))]
		fp_tier.add_annotations(fp_intervals)
		tg.add_tier(fp_tier)

	else:

		word_durations = []
		for w in word_tier._get_annotations():
			word_durations.append(w)

		bp = ['start']+['<unk>' for i in range(len(word_durations)-1)]
		fp = ['<unk>' for i in range(len(word_durations)-1)]+['end']

		bp_tier = tgt.IntervalTier()
		bp_tier.name = [name for name in tier_names if 'words' in name][0].replace('words', 'bp')
		bp_intervals = [tgt.Interval(word_durations[i].start_time, word_durations[i].end_time, bp[i]) for i in range(0, len(word_durations))]
		bp_tier.add_annotations(bp_intervals)
		tg.add_tier(bp_tier)

		fp_tier = tgt.IntervalTier()
		fp_tier.name = [name for name in tier_names if 'words' in name][0].replace('words', 'fp')
		fp_intervals = [tgt.Interval(word_durations[i].start_time, word_durations[i].end_time, fp[i]) for i in range(0, len(word_durations))]
		fp_tier.add_annotations(fp_intervals)
		tg.add_tier(fp_tier)

	# For now we generate the modified TextGrids in the same folder is the old ones. Later, sent the new files into a new folder
	tgt.write_to_file(tg, os.path.join(output_path,title+'.TextGrid'), format='short')

if __name__ == '__main__':

	textgrid_path = sys.argv[1]
	txt_path = sys.argv[2]
	output_path = sys.argv[3]

	#textgrid_path = '../build/11_textgrid_lemmas'
	#txt_path = '../build/05_txt'
	#output_path = '../build/12_textgrid_punctuation'

	textgrid_titles = load_titles(textgrid_path, '.TextGrid')
	txt_titles = load_titles(txt_path, '.txt')

	titles = list(set(txt_titles).intersection(textgrid_titles))

	p = Pool()
	p.starmap(add_punctuation, [(title, textgrid_path, txt_path, output_path) for title in titles])

