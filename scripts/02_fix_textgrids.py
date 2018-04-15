import xml.etree.ElementTree as ET
from multiprocessing import Pool
import tgt
import sys
import os


def annotate(title, xml_path, textgrid_path, annotations_path):
	try:


		tree = ET.parse(os.path.join(xml_path,title+'.xml'))
		root = tree.getroot()



		stress_phone_seq = [] # The content here come from the xml file. Output format:  [[ph1, ph2, etc.], [ph1, ph2, etc.], etc]
		stress_seq = [] # The content here come from the xml file. Output format:  [[0], [2], [1],etc]
		for p in root[0]:
			for s in p:
				for phrase in s:
					for word in phrase:
						# get rid of words in xml that lack a phonemic counterpart in the textGrid
						if word.text not in ('!', ',', '-', '.', '..', '...',':', '?'):
							for syllable in word:
								stress_phone_group = []
								stress_group = []
								stress_group.append(syllable.attrib['stress'])
								stress_seq.append(stress_group)
								for ph in syllable:
									stress_phone_group.append(ph.attrib['p'])
								stress_phone_seq.append(stress_phone_group)

		tg = tgt.read_textgrid(os.path.join(textgrid_path,title+'.TextGrid'))
		phones_tier = tg.get_tier_by_name('phones')
		word_tier = tg.get_tier_by_name('words')

		#word_durations = [w for w in word_tier._get_annotations() if w.text != '-'] # use this instead of the next snippet if you remove '-' from the vocabulary. Atm '-' is mapped to 'min@s'
		word_durations = []
		dash_intervals = []
		for w in word_tier._get_annotations():
			if w.text == '-':
				dash_intervals.append(w)
			else:
				word_durations.append(w)
		for dash in dash_intervals:
			# Here we delete all the phone annotation that are read out as "minus", if you don't u mess up the alignment 
			phones_tier.delete_annotations_between_timepoints(dash.start_time, dash.end_time, left_overlap=False, right_overlap=False)


		phone_durations = [p for p in phones_tier._get_annotations() if p.text != 'sil']

		# here we gather the phone durations following the same format as pos_phone_seq, i.e. [[ph_dur1, ph_dur2, etc.], [ph_dur1, ph_dur2, etc.], etc]

		#print([j for i in stress_phone_seq for j in i])
		#print([i.text for i in phone_durations])

		l = []
		k = -1
		for i in range(0, len(stress_phone_seq)):
			m = []
			for j in range(0, len(stress_phone_seq[i])):
				k += 1
				m.append(phone_durations[k])
			l.append(m)


		# here we go thru this list ([[ph_dur1, ph_dur2, etc.], [ph_dur1, ph_dur2, etc.], etc]) and we keep the first and the last duration of every syllable
		syl_durations = [(syl[0].start_time, syl[-1].end_time) for syl in l]
		syllable_tier = tgt.IntervalTier()
		syllable_tier.name = 'syllables'
		syllable_tier.start_time = phones_tier.start_time
		syllable_tier.end_time = phones_tier.end_time
		syllable_intervals = [tgt.Interval(syl_durations[i][0], syl_durations[i][1], str(stress_seq[i][0])) for i in range(0, len(syl_durations))]
		syllable_tier.add_annotations(syllable_intervals)




		for phone in phones_tier:
			phone.text = phone.text.replace('Q', '@@').replace('ts', 't').replace('sp', 'sil')

		vowels = ['@', '@@', 'a', 'aa', 'ai', 'au', 'e', 'e@', 'ei', 'i', 'i@', 'ii', 'o', 'oi', 'oo', 'ou', 'u', 'u@', 'uh', 'uu']

		for phone in phones_tier:
			if phone.text in vowels:

				phone_centre = phone.start_time+(phone.end_time - phone.start_time)/2
				phone.text = phone.text+syllable_tier.get_annotations_by_time(phone_centre)[0].text
				

		# For now we generate the modified TextGrids in the same folder is the old ones. Later, sent the new files into a new folder
		newTitle = os.path.join(annotations_path, title + '.TextGrid')
		tgt.write_to_file(tg, newTitle, format='short')
	except:
		pass

if __name__ == '__main__':
	
	#Arguments needed by the script
	xml_path = sys.argv[1]
	textgrid_path = sys.argv[2]
	annotations_path = sys.argv[3]

	#print(xml_path)
	#print(textgrid_path)
	#print(annotations_path)

	#xml_path = '../build/xml/'
	#textgrid_path = '../build/textgrid/'
	#annotations_path = '../build/annotations/'

	xml_titles = []
	for fn in os.listdir(xml_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.xml':
			xml_titles.append(basename)

	textgrid_titles = []
	for fn in os.listdir(textgrid_path):
		basename, extension = os.path.splitext(fn)
		if extension == '.TextGrid':
			textgrid_titles.append(basename)

	titles = list(set(xml_titles).intersection(textgrid_titles))

	p = Pool()
	p.starmap(annotate, [(title, xml_path, textgrid_path, annotations_path) for title in titles])

