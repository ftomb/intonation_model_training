from multiprocessing import Pool
import subprocess
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

def merge_labels(title, input1_path, input2_path, output_path):

	with open(os.path.join(input1_path,title+'.json')) as f:    
		linguistic_labels = json.load(f)

	with open(os.path.join(input2_path,title+'.json')) as f:    
		f0_labels = json.load(f)

	merged_labels = [linguistic_labels[i]+f0_labels[i] for i in range(len(linguistic_labels))]

	with open(os.path.join(output_path,title+'.json'), 'w') as f:
		json.dump(merged_labels, f)


if __name__ == '__main__':


	input1_path = sys.argv[1]
	input2_path = sys.argv[2]
	output_path = sys.argv[3]

	#input1_path = '../build/17_linguistic_labels'
	#input2_path = '../build/18_f0_labels'
	#output_path = '../build/19_merged_labels'

	titles1 = load_titles(input1_path, '.json')
	titles2 = load_titles(input2_path, '.json')

	titles = list(set(titles1).intersection(titles2))
	
	p = Pool()
	p.starmap(merge_labels, [(title, input1_path, input2_path, output_path) for title in titles])
