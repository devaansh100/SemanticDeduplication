import numpy as np
import glob
import json
from tqdm import tqdm

def get_conll_data(conll_folder):
	entities = []
	files = glob.glob(f'{conll_folder}/*conll')
	for file in files:
		try:
			f = open(file)
		except:
			print(f'Error opening {file}')
			continue
		entities.append([])
		for line in f.readlines():
			if len(line):
				try:
					word, entity = line.replace('\n', '').split(' ')
				except:
					continue					
				if entity != 'O':
					entities[-1].append(word) # NOTE: Not saving the kind of entity right now. Forces T5 to figure out the entity. We might get better results by talking about the kind of entity
		entities[-1] = ' '.join(entities[-1])

	return entities, files


def get_json_data(json_folder, files = None):
	data = []
	files = glob.glob(f'{json_folder}/*json') if files is None else files
	for file in files:
		try:
			f = open(file.split('.')[0] + '.json')
		except:
			print(f'Error opening {file}')
			continue
		data.append(json.loads(f.read()))
	
	return data


def get_raw_data(txt_folder, files = None):
	data = []
	files = glob.glob(f'{json_folder}/*txt') if files is None else files
	for file in files:
		try:
			f = open(file.replace('CONLL', 'TEXT').replace('conll', 'txt'))
		except:
			print(f'Error opening {file}')
			continue
		data.append(f.read())

	return data

def tokenize(text_list, tokenizer):
	return [tokenizer.encode_plus(text, add_special_tokens = True, max_length = 512, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt') for text in text_list]
