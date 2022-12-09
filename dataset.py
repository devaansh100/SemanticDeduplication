from torch.utils.data import Dataset
import random
import torch

class WildlifeDataset(Dataset):
	def __init__(self, entities, articles, tokenizer, params):
		super().__init__()
		self.entities = entities
		self.articles = articles
		self.seq_len = params.seq_len

	def __len__(self):
		return len(self.articles)

	def __getitem__(self, idx):
		entity = self.entities[idx]
		article = self.articles[idx]
		return entity, article


class ContrastiveDataset(Dataset):
	def __init__(self, anchors, positives, tokenizer, params):
		super().__init__()
		self.anchors = anchors
		self.positives = positives
		self.seq_len = params.seq_len
		self.pos_method = params.positive_method

	def __len__(self):
		return len(self.anchors)

	def __getitem__(self, idx):
		anchor = self.anchors[idx]
		if self.pos_method == 'generate':
			pos = self.positives[idx]
		else:
			first_non_zero = torch.nonzero(anchor['input_ids'][0] == 0).squeeze()
			first_non_zero = len(anchor['input_ids'][0]) if len(first_non_zero) == 0 else first_non_zero[0]
			pos_input_ids = anchor['input_ids'][0][:first_non_zero]
			drop_mask = torch.FloatTensor(pos_input_ids.shape).uniform_() > 0.5
			pos_input_ids = pos_input_ids * drop_mask
			pos_input_ids = pos_input_ids[pos_input_ids > 0]
			pos_att_mask = torch.cat((torch.ones_like(pos_input_ids), torch.zeros(len(anchor['input_ids'][0]) - len(pos_input_ids))), dim = 0)
			pos_input_ids = torch.cat((pos_input_ids, torch.zeros(len(anchor['input_ids'][0]) - len(pos_input_ids))), dim = 0).long()
			pos = {'input_ids': pos_input_ids.unsqueeze(0), 'attention_mask': pos_att_mask.unsqueeze(0)}

		return anchor, pos

