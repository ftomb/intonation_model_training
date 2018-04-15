from multiprocessing import Pool
import numpy as np
import subprocess
import string
import shutil
import wave
import math
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

def sort_titles(train_titles):

	book_titles = {}
	for title in train_titles:
		book_title = ''.join(['0' if char in string.digits else char for char in title])
		book_title = book_title.split('0')
		if book_title[0] not in book_titles.keys():
			book_titles[book_title[0]] = [title]
		else:
			book_titles[book_title[0]].append(title)

	return book_titles


def stitch_txt(batch_title, sequenced_title, input1_path, output2_path):

	combined_txt = []

	for title in sequenced_title:
		with open(os.path.join(input1_path,title+'.txt')) as f:
			combined_txt+=[l.strip() for l in f]

	with open(os.path.join(output2_path,batch_title+'.txt'), 'w') as g:
		g.write(' '.join(combined_txt))	

def stitch_f0(batch_title, sequenced_title, input2b_path, output1_path):

	combined_f0s = []
	last_dur = 0.0
	for title in sequenced_title:
		with open(os.path.join(input2b_path,title+'.f0')) as f:
			durs = []
			f0s = []
			for l in f:
				dur, f0 = l.strip().split()
				durs.append(float(dur))
				f0s.append(float(f0))
			durs = durs[:-1]
			f0s = f0s[:-1]
		durs = [dur+last_dur for dur in durs]
		last_dur = round(durs[-1]+0.005, 3)
		for i, dur in enumerate(durs):
			combined_f0s.append(str(durs[i])+'\t'+str(f0s[i]))

	with open(os.path.join(output1_path,batch_title+'.f0'), 'w') as g:
		g.write('\n'.join(combined_f0s))	

def stitch_wav(batch_title, sequenced_title, input2_path, output1_path):

	combined_wav = bytearray()

	for title in sequenced_title:

		wave_file = wave.open(os.path.join(input2_path, title+'.wav'), 'rb')
		params = wave_file.getparams()
		n_channels = wave_file.getnchannels()
		frameRate = wave_file.getframerate()
		n_frames = wave_file.getnframes()
		wave_slice = wave_file.readframes(n_frames)
		combined_wav.extend(wave_slice)
		dur = n_frames/frameRate

	wave_slice_file = wave.open(os.path.join(output1_path, batch_title+'.wav'), 'wb')
	wave_slice_file.setparams(params)
	wave_slice_file.writeframes(combined_wav)

def stitch_textgrid(batch_title, sequenced_title, input2b_path, input2_path, output3_path):		
	combined_intervals = []

	new_tg = tgt.TextGrid()

	new_phone_tier = tgt.IntervalTier()
	final_phone_tier = tgt.IntervalTier()
	new_word_tier = tgt.IntervalTier()


	last_dur = 0.0


	for i, title in enumerate(sequenced_title):

		wave_file = wave.open(os.path.join(input2b_path, title+'.wav'), 'rb')
		frameRate = wave_file.getframerate()
		n_frames = wave_file.getnframes()
		dur = n_frames/frameRate

		f0_start_time = 0.0
		f0_end_time = dur

		tg = tgt.read_textgrid(os.path.join(input2_path,title+'.TextGrid'))

		# Load name of all tiers
		tier_names = tg.get_tier_names()

		words_tier_name = [name for name in tier_names if 'words' in name][0]
		words_tier = tg.get_tier_by_name(words_tier_name)

		phones_tier_name = [name for name in tier_names if 'phones' in name][0]
		phones_tier = tg.get_tier_by_name(phones_tier_name)

		word_annotations = words_tier.get_annotations_between_timepoints(f0_start_time, f0_end_time)
		phone_annotations = phones_tier.get_annotations_between_timepoints(f0_start_time, f0_end_time)

		word_intervals = []
		for interval in word_annotations:
			interval.end_time = interval.end_time+last_dur
			interval.start_time = interval.start_time+last_dur
			word_intervals.append(interval)
		if word_intervals[-1].end_time > last_dur+f0_end_time:
			word_intervals[-1].end_time = last_dur+f0_end_time


		phone_intervals = []
		for j, interval in enumerate(phone_annotations):
			interval.end_time = interval.end_time+last_dur
			interval.start_time = interval.start_time+last_dur

			if interval.text != 'sil' and interval.text != 'sp':
				phone_intervals.append(interval)

			elif i == len(sequenced_title)-1 and j == len(phone_annotations)-1:
				phone_intervals.append(interval)
		if phone_intervals[-1].end_time > last_dur+f0_end_time:
			phone_intervals[-1].end_time = last_dur+f0_end_time

		new_word_tier.add_annotations(word_intervals)
		new_phone_tier.add_annotations(phone_intervals)

		last_dur += dur

	phones_tier_copy = new_phone_tier.get_copy_with_gaps_filled(start_time=None, end_time=None, empty_string='')


	# Replace all sil and sp intervals with <sil> tag 
	#store these intervals to a list so that we can add them to the other tiers
	sil_intervals = []
	phone_intervals = []
	for interval in phones_tier_copy:
		if interval.text == '':
			interval.text = 'sil'
			sil_intervals.append(interval)
		else:
			phone_intervals.append(interval)

	final_phone_tier.add_annotations(phone_intervals)
	final_phone_tier.add_annotations(sil_intervals)

	final_phone_tier.name = phones_tier_name
	new_word_tier.name = words_tier_name

	new_tg.add_tier(new_word_tier)
	new_tg.add_tier(final_phone_tier)

	tgt.write_to_file(new_tg, os.path.join(output3_path,batch_title+'.TextGrid'), format='short')



def stich_sequencial_titles(train_titles, kernel):
	titles = []
	if kernel > len(train_titles):
		kernel = len(train_titles)
	if kernel < 2:
		kernel = 2
	for i in range(0, len(train_titles[:-kernel])+1, kernel):
		titles += [train_titles[i:i+kernel]]
	return titles


def stich_random_titles(train_titles, pop_size, n_extractions):
	np.random.seed(seed=0)

	# pop_size is how many titles we want to stich together
	random_titles = []
	# n_extractions is how many times we want to draw random titles from the population
	for i in range(n_extractions):
		current_random_titles = list(np.random.choice(train_titles, size=pop_size))
		random_titles.append(current_random_titles)

	return random_titles

def generate_title_batches(book_titles, max_length):
	sequenced_titles = []
	for book_title in list(book_titles.keys()):
		for kernel in range(1, max_length):
			sequenced_titles += stich_sequencial_titles(book_titles[book_title], kernel)
	# Remove duplicate sequences
	b_set = set(map(tuple,sequenced_titles))
	b = map(list,b_set) 
	sequenced_titles = [item for item in b]


	return sequenced_titles

def generate_random_title_batches(train_titles, max_length):

	random_titles = []
	for length in range(8, max_length+1):
		'''
		if length == 2:
			random_titles += stich_random_titles(train_titles, length, len(train_titles))
		else:
			random_titles += stich_random_titles(train_titles, length, round(len(train_titles)/max_length))
		'''
		random_titles += stich_sequencial_titles(train_titles, length)

	# Remove duplicate sequences
	#b_set = set(map(tuple,random_titles))
	#b = map(list,b_set) 
	#random_titles = [item for item in b]

	return random_titles

def augment_data(titles, input1_path, input2_path, input2b_path, output1_path, output2_path, output3_path, max_length, input3_path, titles_json):

	#book_titles = sort_titles(titles)

	sequenced_titles = generate_random_title_batches(titles, max_length)
	print(len(sequenced_titles))

	counter = 0

	for sequenced_title in sequenced_titles:

		fill_amount = len(str(len(sequenced_titles)))
		subtitle = str(counter+1).zfill(fill_amount)

		counter += 1
		batch_title = '_aug_'+subtitle

		titles_json['train'].append(batch_title)

		stitch_txt(batch_title, sequenced_title, input1_path, output2_path)
		stitch_wav(batch_title, sequenced_title, input2b_path, output1_path)
		#stitch_f0(batch_title, sequenced_title, input2b_path, output1_path)
		stitch_textgrid(batch_title, sequenced_title, input2b_path, input2_path, output3_path)

	with open(os.path.join(input3_path, 'augmented_split_titles.json'), 'w') as f:    
		json.dump(titles_json, f)

def copy_to_corpus(title, from_path, to_path, extension):
	shutil.copy(os.path.join(from_path,title+extension), os.path.join(to_path,title+extension))


def copy_files(titles, from_path, to_path, extension):
	p = Pool()
	p.starmap(copy_to_corpus, [(title, from_path, to_path, extension) for title in titles])

def copy_data(titles, input1_path, input2_path, input2b_path, output1_path, output2_path, output3_path, extension1, extension2, extension3):
	copy_files(titles, input1_path, output2_path, extension1)
	copy_files(titles, input2_path, output3_path, extension2)
	copy_files(titles, input2b_path, output1_path, extension3)


if __name__ == '__main__':


	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	input2b_path = sys.argv[3]
	input3_path = sys.argv[4]
	output1_path = sys.argv[5]
	output2_path = sys.argv[6]
	output3_path = sys.argv[7]

	#input1_path = '../build/00_txt'
	#input2_path = '../build/00_textgrid'
	#input2b_path = '../build/00_wav'
	#input3_path = '../build/03_split_test_train_valid'
	#output1_path = '../build/05_wav'
	#output2_path = '../build/04_txt'
	#output3_path = '../build/03_textgrid'

	with open(os.path.join(input3_path, 'split_titles.json')) as f:    
		titles_json = json.load(f)
	train_titles = titles_json['train']

	txt_titles = load_titles(input1_path, '.txt')
	textgrid_titles = load_titles(input2_path, '.TextGrid')
	wav_titles = load_titles(input2b_path, '.wav')

	augment_titles = sorted(list(set(train_titles).intersection(txt_titles).intersection(textgrid_titles).intersection(wav_titles)))
	all_titles = sorted(list(set(txt_titles).intersection(textgrid_titles).intersection(wav_titles)))

	# This is how many titles at most we want to combine
	max_length = 16

	# only augment TRAIN data
	augment_data(augment_titles, input1_path, input2_path, input2b_path, output1_path, output2_path, output3_path, max_length, input3_path, titles_json)

	# copy all the data, but later only augmented and train data are used for training
	copy_data(all_titles, input1_path, input2_path, input2b_path, output1_path, output2_path, output3_path, '.txt', '.TextGrid', '.wav')