from multiprocessing import Pool
import subprocess
import random
import string
import wave
import math
import json
import sys
import os

def load_titles(target_path, extension):
	titles = []
	for fn in os.listdir(target_path):
		basename, ext = os.path.splitext(fn)
		if ext == extension:
			titles.append(basename)
	return titles

def sort_titles(titles):

	book_titles = {}
	for title in titles:
		book_title = ''.join(['0' if char in string.digits else char for char in title])
		book_title = book_title.split('0')
		if book_title[0] not in book_titles.keys():
			book_titles[book_title[0]] = [title]
		else:
			book_titles[book_title[0]].append(title)

	return [book_titles[titles] for titles in book_titles.keys()]

def calculate_durations(titles, input2_path):

	durations = {}

	for title in titles:
		wave_file = wave.open(os.path.join(input2_path, title+'.wav'), 'rb')
		frameRate = wave_file.getframerate()
		n_frames = wave_file.getnframes()
		duration = n_frames/frameRate
		durations[title] = duration

	return durations

def calculate_upper_lower_dur(titles, input2_path):

	durations = calculate_durations(titles, input2_path)
	
	mean_dur = sum(durations.values())/len(durations.values())
	std = math.sqrt(sum((v-mean_dur)**2 for v in durations.values())/len(durations.values()))

	upper_dur = mean_dur+std
	lower_dur = mean_dur-std

	return lower_dur, mean_dur, upper_dur, durations	

def filter_books_out(titles, input2_path, cutoff_ratio): 

	lower_dur, mean_dur, upper_dur, durations= calculate_upper_lower_dur(titles, input2_path)

	# remove first and last n% of book
	books = sort_titles(titles)
	cutoff = lambda x: math.ceil(len(x)*cutoff_ratio)
	middle_titles = [title for book in books for title in book[cutoff(book):-cutoff(book)]]

	middle_avg_dur_titles = [book for book in middle_titles if durations[book]>lower_dur and durations[book]<upper_dur] 

	return middle_avg_dur_titles


if __name__ == '__main__':


	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	input3_path = sys.argv[3]
	output_path = sys.argv[4]

	#input1_path = '../build/00_txt/'
	#input2_path = '../build/00_wav/'
	#input3_path = '../build/00_textgrid/'
	#output_path = '../build/06_train_test_titles'

	txt_titles = load_titles(input1_path, '.txt')
	wav_titles = load_titles(input2_path, '.wav')
	textgrid_titles = load_titles(input3_path, '.TextGrid')

	titles = sorted(list(set(txt_titles).intersection(wav_titles).intersection(textgrid_titles)))

	# exclude titles that are not in the middle
	# exclude titles that are outside the standard deviation of duration
	candidates = filter_books_out(titles, input2_path, 0.1)
	random.seed(a=0, version=2)
	random.shuffle(candidates)

	test_ratio = 0.1
	test_split_index = math.ceil(len(titles)*test_ratio)

	if test_split_index < len(candidates):	
		test_titles = candidates[:test_split_index]
	else:
		test_titles = candidates
		print('Not enough titles!')

	candidates = [title for title in candidates if title not in test_titles]
	random.shuffle(candidates)

	valid_ratio = 0.05
	valid_split_index = math.ceil(len(candidates)*valid_ratio)

	if valid_split_index < len(candidates):	
		valid_titles = candidates[:valid_split_index]
	else:
		valid_titles = candidates
		print('Not enough titles!')

	train_titles = [title for title in titles if title not in valid_titles and title not in test_titles]
	split_titles = {'test':sorted(test_titles), 'train':sorted(train_titles), 'valid':sorted(valid_titles)}

	with open(os.path.join(output_path,'split_titles.json'), 'w') as f:
		json.dump(split_titles, f)
