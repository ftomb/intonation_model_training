from scipy import interpolate as interp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.io import wavfile
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

def load_f0_timepoints(path, title):
    with open(os.path.join(path,title+'.json')) as f:    
        return json.load(f)

def import_f0_data(path,title):
    with open (os.path.join(path,title+'.f0')) as f:
        f0 = [float(l.split()[1]) for l in f]
        log_f0 = np.log2(f0)
        return f0, log_f0

def convert_to_herz(f0s):
    return np.exp2(f0s)

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

def f0_encoding(s, log_f0_seq):
    '''This encodes the f0 sequence into a sequence of semitone intervals'''

    # Calculate average
    log_f0_avg = sum(log_f0_seq)/len(log_f0_seq)

    list_pitches = generate_scale(s)

    # Find the pitch in the scale that is closest to the first f0 value 
    seed_deltas = [abs(log_f0_seq[0] - list_pitches[j]) for j in range(0, len(list_pitches))]

    # Store its index as a seed
    index_seed = seed_deltas.index(min(seed_deltas))
    #print(seed_deltas)

    # Initialize a list to store the semintone steps (e.g. "-1", "4", etc.)
    step_seq = []

    # Initialize a list with the first log2 f0 value
    log_f0_seq_ = log_f0_seq[0:1]
    
    # In the next loop we populate the log_f0_seq_ list
    for i in range(0, len(log_f0_seq)-1):

        # For each f0 we generate a scale. The zero of the scale is set to correspond to the current f0 e.g. [[-1, f0], [0, current_f0], [1, f0], [3, f0], etc.]
        scale = generate_tri_scale(s, 2**log_f0_seq_[-1])

        # compare each note in the scale with the next log_f0 value and store it in a list along with the index (note[0]) if these comparisons 
        deltas = [[note[0], note[1], abs(note[1]-log_f0_seq[i+1])] for note in scale]

        # Order the deltas based on the distance from the next log_f0 value
        # Step tells you how many semitones are required to reach the next f0_log value
        # f0_update replaces the next f0_log with the value we would get by as many semitones in the step
        # min_delta is the distance between the f0 we get by perfoming the jump and the actual next log_f0 value
        step, f0_update, min_delta = min(deltas, key = lambda t: t[2])
        #print(f0_update)

        step_seq.append(step)
        log_f0_seq_.append(f0_update)

    # step_seq_x is needed as the input of the neural network 
    # step_seq_y is needed as the output of the neural network
    # We use the same sequenece shifted in order to have a recurrently feeding output
    step_seq_x = [0]+step_seq
    step_seq_y = step_seq+[0]
    #print(log_f0_avg)

    return log_f0_avg, step_seq_x, step_seq_y
    
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


def extract_f0_labels(title, input1_path, input2_path, output_path, s):
    try:
        # Sampling rate at which f0 were extracted from original recording
        audio_sr = 0.005
        # Load raw f0 measurements from txt file
        raw_f0s, raw_log_f0s = import_f0_data(input1_path,title)

        # Smooth the f0 sequence
        smoothed_raw_log_f0s = smooth_f0(raw_log_f0s, 31, 2)

        # Load the f0_timepoints (where linguistic labels are also extracted)
        f0_timepoints = load_f0_timepoints(input2_path,title)

        # For each timepoint (location where linguistic labels are sampled) extracted its corresponding f0
        sampled_log_f0s = [smoothed_raw_log_f0s[round(t/audio_sr)] for t in f0_timepoints]

        # Convert the sequence of sampled f0s into a sequence of semitone steps
        log_f0_avg, step_seq_x, step_seq_y = f0_encoding(s, sampled_log_f0s)

        f0_labels = [[str(step_seq_x[i])]+[str(step_seq_y[i])]+[str(np.sign(step_seq_x[i]))]+[str(np.sign(step_seq_y[i]))]+[str(np.absolute(step_seq_x[i]))]+[str(np.absolute(step_seq_y[i]))] for i in range(len(step_seq_x))]

        with open(os.path.join(output_path,title+'.json'), 'w') as f:
            json.dump(f0_labels, f)
    except:
        print(title)
        pass


if __name__ == '__main__':

    input1_path = sys.argv[1]
    input2_path = sys.argv[2]
    output_path = sys.argv[3]

    #input1_path = '../build/15_augmented_f0'
    #input2_path = '../build/16_f0_timepoints'
    #output_path = '../build/18_f0_labels'

    # This parameter is the frequency resolution, i.e. vertical sampling rate
    s = 24

    titles1 = load_titles(input1_path, '.f0')
    titles2 = load_titles(input2_path, '.json')

    titles = list(set(titles1).intersection(titles2))

    p = Pool()
    p.starmap(extract_f0_labels, [(title, input1_path, input2_path, output_path, s) for title in titles])
