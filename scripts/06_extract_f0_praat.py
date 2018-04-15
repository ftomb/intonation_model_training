from multiprocessing import Pool
import subprocess
import sys
import os


def load_titles(target_path, extension):
	titles = []
	for fn in os.listdir(target_path):
		basename, ext = os.path.splitext(fn)
		if ext == extension:
			titles.append(basename)
	return titles

def extract_f0(title, praat_path, praat_script, wav_path, inter_f0_path):

	wav_path = os.path.join(wav_path, title+'.wav')
	inter_f0_path = os.path.join(inter_f0_path, title+'.f0')
	subprocess.call([praat_path, '--run', praat_script, wav_path,inter_f0_path])

if __name__ == '__main__':

	wav_path = sys.argv[1]
	inter_f0_path = sys.argv[2]
	praat_path = sys.argv[3]
	praat_script = sys.argv[4]

	#wav_path = '../build/02_wav/'
	#inter_f0_path = '../build/03_f0/'

	#praat_path = '../build/praat/praat'
	#praat_script = 'praat_script.praat'

	titles = load_titles(wav_path, '.wav')

	p = Pool()
	p.starmap(extract_f0, [(title, praat_path, praat_script, wav_path, inter_f0_path) for title in titles])

	

	