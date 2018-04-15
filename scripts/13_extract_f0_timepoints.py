from multiprocessing import Pool
import matplotlib.pyplot as plt
import subprocess
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

def extract_f0_timepoints(title, input_path, output_path, f0_sr):

	tg = tgt.read_textgrid(os.path.join(input_path,title+'.TextGrid'))

	# Original recording's sampling rate
	audio_sr = 0.005 

	# Load name of all tiers
	tier_names = tg.get_tier_names()

	# Select a tier whose name contains 'sylls'
	syllables_tier_name = [name for name in tier_names if 'sylls' in name][0]
	syllables_tier = tg.get_tier_by_name(syllables_tier_name)

	# In this list we collect the timepoints where we sample the f0
	f0_timepoints = [] 

	# Sampling is syllable based, so we loop over each syllable
	for interval in syllables_tier:

		# add the first time of the syllable to the list
		interval_start = interval.start_time
		f0_timepoints.append(float(interval_start))

		# plot the syllable boundary
		#plt.axvline(x=interval_start, color='orange', linewidth=1)

		# How many times the default sampling rate approx. fits into this interval
		n_extractions = round(interval.duration()/f0_sr)

		# This happens if the interval is shorter than the sampling rate, so we just keep the first value for this interval
		if n_extractions == 0:
			pass

		else:
			# Based on the N of extractions calculate a new sampling rate, which is specific to this interval
			sampling_step = interval.duration()/(n_extractions)

			# Extract based on the new sampling rate 
			# The loop is for n_extractions-1 times because the last point is the first point of the next interval
			for j in range(0, n_extractions-1):
				interval_start += sampling_step
				f0_timepoints.append(interval_start)
				
				# Plot extraction points between boundaries
				#plt.axvline(x=interval_start, color='gray', linewidth=0.5)

	# Add the very last time of the last syllable
	interval_start = float(syllables_tier[-1].end_time)
	f0_timepoints.append(interval_start)

	# Plot the last point of the last syllable
	#plt.axvline(x=interval_start, color='orange', linewidth=1)

	#plt.show()

	with open(os.path.join(output_path,title+'.json'), 'w') as f:
		json.dump(f0_timepoints, f)


if __name__ == '__main__':

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	#input_path = '../build/14_augmented_texgrid'
	#output_path = '../build/16_f0_timepoints'

	# How often we sample the f0
	f0_sr = 0.1

	titles = load_titles(input_path, '.TextGrid')

	p = Pool()
	p.starmap(extract_f0_timepoints, [(title, input_path, output_path, f0_sr) for title in titles])
