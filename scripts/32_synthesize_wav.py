from multiprocessing import Pool
from scipy.io import wavfile
import pyworld as pw 
import numpy as np
import subprocess
import pysptk
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

def import_f0_data(path,title):
	#return np.fromfile(file_name, dtype=np.float32)
	with open (os.path.join(path,title+'.f0')) as f:
		return np.array([float(l.strip()) for l in f], dtype=np.float64)

def load_wav(path,title):
	wav_stream = wavfile.read(os.path.join(path,title+'.wav'))
	return np.array(np.array(wav_stream[1], dtype=float), dtype=np.float64)

def vocoder_synthesis(title, input1_path, input2_path, subtitle, output_path, fs):

	print('Synthesizing: ', title)

	synth_f0 = import_f0_data(input1_path, title)
	x = load_wav(input2_path,title)

	_f0, ts = pw.dio(x, fs)    # raw pitch extractor

	f0 = pw.stonemask(x, _f0, ts, fs)  # pitch refinement
	sp = pw.cheaptrick(x, f0, ts, fs)  # extract smoothed spectrogram
	ap = pw.d4c(x, f0, ts, fs)         # extract aperiodicity

	sp = sp[:len(synth_f0)]
	ap = ap[:len(synth_f0)]

	y = pw.synthesize(synth_f0, sp, ap, fs)

	wavfile.write(os.path.join(output_path,title+'_'+subtitle+'.wav'), fs, np.array(y, dtype=np.int16))
	print(title)


if __name__ == '__main__':

	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	subtitle = sys.argv[3]
	output_path = sys.argv[4]

	#input1_path = '../build/38_synth_f0s'
	#input2_path = '../build/08_wav'
	#output_path = '../build/39_synth_wav'

	fs = 48000

	titles1 = load_titles(input1_path, '.f0')
	titles2 = load_titles(input2_path, '.wav')

	titles = sorted(list(set(titles1).intersection(titles2)))

	p = Pool()
	p.starmap(vocoder_synthesis, [(title, input1_path, input2_path, subtitle, output_path, fs) for title in titles])



