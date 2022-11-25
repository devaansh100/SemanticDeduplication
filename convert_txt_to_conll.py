import numpy as np
import glob
from tqdm import tqdm
import re

conll_folder = 'Annotated_Articles/Final_CONLL/'
txt_folder = 'Annotated_Articles/Final_TEXT'
files_txt   = glob.glob(f'{txt_folder}/*txt')
num_issues = 0
for file in tqdm(files_txt):
	with open(file) as f:
		contents = f.read()
		try:
			if '[' in contents and ']' in contents:
				article, entities = contents.split("']")
				article = article.replace("['", '').replace('\n', '')
				nes, lbls = [], []
				for entity in entities.split('\n\n'):
					if '----' in entity:
						ne, lbl = entity.split('----')
						nes.append(ne.replace('\n', ''))
						lbls.append(lbl.replace('\n', ''))
				words = []
				tags = []
				idx, word_idx = 0, 0
				# Iterate through each word in the article
				# It is guaranteed that the entities are in sequenial order
				# When you encounter the entity in the word from article, add that tag, else add 0
				# If it is a multi-word entity, add the same tag for all the words and then increment 
				# the word_idx till all the words are covered. Only increment the tag idx once the entity words are covered
				# TODO: Break the punctuation into different tokens as well
				for word in article.split(' '):
					if idx < len(nes):
						entity_words = nes[idx].split(' ')
					else:
						entity_words = []
					words.append(word)
					if len(entity_words) > 0 and entity_words[word_idx] in word:
						tags.append(lbls[idx])
						word_idx += 1
						if word_idx == len(entity_words):
							idx += 1
							word_idx = 0
					else:
						tags.append('0')
				write_file = conll_folder + file.split('/')[-1].replace('.txt', '.conll')
				with open(write_file, 'w') as f_w:
					for w, ne in zip(words, tags):
						f_w.write(f'{w} {ne}\n')
		except:
			num_issues += 1
			continue
	with open(file, 'w') as f:
		f.write(' '.join(words))

print(f'Completed. Issue in converting {num_issues} files')


files_conll = glob.glob(f'{conll_folder}/*conll')
fc = set()
ft = set()

for f in files_conll:
	fc.add(f.split('/')[-1].replace('.conll','.txt'))

for f in files_txt:
	ft.add(f.split('/')[-1])

print(f'Total Samples = {len(fc.intersection(ft))}')