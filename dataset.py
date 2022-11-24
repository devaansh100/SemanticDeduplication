from torch.utils.data import Dataset
import random
import torch

class WildlifeDataset(Dataset):
	def __init__(self, entities, articles, tokenizer, params):
		super().__init__()
		self.entities = entities
		self.articles = articles
		self.subset = params.subset
		self.seq_len = params.seq_len

	def __len__(self):
		return len(self.articles)

	def pad_item(self, item):
		if len(item['input_ids']) > self.seq_len:
			item['input_ids'] = self.cls_token_id + item['input_ids'][:self.seq_len - 1]
		else:
			item['input_ids'] = self.cls_token_id + item['input_ids'] + self.pad_token_id * (self.seq_len - len(item['input_ids']))

		return item

	def __getitem__(self, idx):
		# start_idx = random.choice(range(len(self.articles[idx]) - self.seq_len))
		# span = self.articles[start_idx:start_idx + self.seq_len]
		# entities = self.entities[idx][torch.randperm(len(self.entities[idx]))[0:self.subset]]
		# entities_conc = []
		# for entity in entities:
		# 	entities_conc.append(entity)
		# 	if entity != entities[-1]:
		# 		entities_conc.append(torch.tensor(self.sep_token_id))
		# entities_conc = entities_conc + [torch.tensor([self.pad_token_id]*(self.seq_len - len(entities_conc)))]
		# entities_conc = torch.cat(entities_conc)
		entity = self.entities[idx]
		article = self.articles[idx]
		return entity, article # span, entities_conc