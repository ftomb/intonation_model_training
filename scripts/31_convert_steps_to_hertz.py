from scipy import interpolate as interp
from multiprocessing import Pool
from scipy import signal as sg
import numpy as np
import subprocess
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

def load_data(input_path, title):
	with open(os.path.join(input_path,title+'.json')) as f:   
		return json.load(f)

def save_f0(output_path, title, y):
	with open(os.path.join(output_path,title + '.f0'), 'w') as g:
		for i in range(0, len(y)):
			txt = str(y[i])+ '\n'
			g.write(txt)

def save_f02(output_path, title, y):
	with open(os.path.join(output_path,title + '.f0'), 'w') as g:
		for i in range(0, len(y)):
			txt = str(round(i*0.005, 3))+ '\t'+str(y[i])+ '\n'
			g.write(txt)

def convert_to_herz(f0s):
	return [2**(f0) for f0 in f0s]

def smooth_f0(raw_f0s, wl, po):
	'''where wl is the window length and po is the polyorder'''
	return sg.savgol_filter(raw_f0s, wl, po)
    #return np.exp(np.fromfile(file_name, dtype=np.float32))

def generate_scale(s):
    '''this scale is needed to split the freq domain in descrete steps (freq. resolution)'''
    list_pitches = []
    n = 0 # this is the number of half steps away from the fixed note (f0 variable in our case)
    fn = 0 # this is the fundamental frequency of certain point in the scale
    f_start = 1 # this is the value you want to start from
    f_end = 1024 # this is how high u want ur scale to reach
    a = 2**(1/s) # where s is the number of semitones you want to be within each octave, e.g. 12 would produce the western chromatic scale
    n = 0 # this is the number of half steps away from the fixed note (f0 variable in our case)
    while (fn < f_end): #the limit is the highest f0 u want to be able to capture
        fn = f_start*((a)**n) 
        list_pitches.append(math.log2(fn))
        n += 1
    return list_pitches

def tri_num(n, T = {0: 0}):
	# This function takes in a triangular number and tells you its index within the scale
	# E.g. triangular_seq = [0 1 3 6 10 etc.], if you feed "3" the output is "2", i.e. second element in the list. As for negative numbers produce negative indeces
	if n not in T:
		if n>0:
			T[n] = tri_num(n-1, T) + n
		else:
			T[n] = tri_num(n+1, T) + n
	return T[n]

def generate_tri_scale(s, start):
	'''This calculates semitone scales based triangular number scales'''
	# Format: [[n1, f01], [n2, f02], etc.], where n is the index (i.e. number of semitones) and f0 is its corresponding herz value
	# Functions produces the next 36 triangular numbers above and below zero as indeces and the corresponding f0s as values
	tri_seq = sorted(set([tri_num(i) for i in range(36)]+[-tri_num(i) for i in range(36)]))
	return [[n, math.log2(start*((2**(1/s))**n))] for n in tri_seq]

def f0_decoding(s, log_f0_avg, step_seq):

    ############# THIS PART DECODES THE FREQUENCY STEPS BACK INTO F0 VALUES ##############
    # The decoding needs the encoded sequence and sentence's average log_f0
    # step_seq here is the same as step_seq_x of the previous encoding function
    
    # Load the reference scale
    list_pitches = generate_scale(s)

    # Seed to start conversion. Use any seed, but choose 0 for convenience
    # The seed doesn't matter because in the end the average determines the actual position in the scale
    index_seed = 0

    # this loop turns a sequence of steps into a sequence of indeces where the first one is 0.
    # E.g. [0, 1, -2, -3] --> [0, 1, -1, -4]
    decoded_step_indices = []
    for i in step_seq:
        index_seed += i
        decoded_step_indices.append(index_seed)
    # Our scale doesn't have negative indices so we make sure all values in the decoding are above zero (added an extra "1" for good measure)
    lowest_index = min(decoded_step_indices)
    if lowest_index<0:
    	decoded_step_indices = [step+abs(lowest_index)+1 for step in decoded_step_indices]

   	# Convert the indeces to actual log_f0 values
    decoded_step_conversion = [list_pitches[i] for i in decoded_step_indices]

    # Calculate the mean of the decoded log_f0 values
    mean = sum(decoded_step_conversion)/len(decoded_step_conversion)

    # Subtract this mean from all the decoded values and add the target average that we want the final sentence to have and convert back to herz
    decoded_step_conversion = [i-mean+log_f0_avg for i in decoded_step_conversion]

    return decoded_step_conversion

def interpolate_f0s(f0_timepoints, log_f0s):
	f = interp.interp1d(f0_timepoints, log_f0s, kind='quadratic')
	x = np.linspace(min(f0_timepoints), max(f0_timepoints), (f0_timepoints[-1]-f0_timepoints[0])/0.005)
	return x, f(x)

def convert_steps_to_herz(title, input1_path, input2_path, output_path, s, log_f0_avg):
	step_seq = load_data(input2_path, title)
	f0_timepoints = load_data(input1_path, title)
	log_f0s = f0_decoding(s, log_f0_avg, step_seq)
	time, log_f0s = interpolate_f0s(f0_timepoints, log_f0s)
	f0s = np.exp2(log_f0s)
	#save_f0(output_path, title, f0s)
	save_f0(output_path, title, f0s)



if __name__ == '__main__':

	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	output_path = sys.argv[3]

	#input1_path = '../build/22_f0_timepoints'
	#input2_path = '../build/37_predicted_f0_steps'
	#output_path = '../build/38_synth_f0s'

	f0_timepoints_titles = load_titles(input1_path, '.json')
	f0_steps_titles = load_titles(input2_path, '.json')

	titles = sorted(list(set(f0_timepoints_titles).intersection(f0_steps_titles)))

	# This parameter is the frequency resolution, i.e. vertical sampling rate
	s = 24

	# This paramenters is what we want the average f0 value that we want for the sentence
	log_f0_avg = 7.5


	p = Pool()
	p.starmap(convert_steps_to_herz, [(title, input1_path, input2_path, output_path, s, log_f0_avg) for title in titles])

