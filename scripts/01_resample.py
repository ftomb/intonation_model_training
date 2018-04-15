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

def resample(title, input_path, output_path, extension):
	subprocess.call(['sox', os.path.join(input_path, title+extension), '-b', '16', os.path.join(output_path, title+extension), 'channels', '1', 'rate', '48000'])

if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	#input_path = '../build/00_wav'
	#output_path = '../build/01_resampled_wav'

	extension = '.wav'
	titles = load_titles(input_path, extension)


	p = Pool()
	p.starmap(resample, [(title, input_path, output_path, extension) for title in titles])