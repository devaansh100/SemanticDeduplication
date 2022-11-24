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

	def __len__(self):
		return len(self.anchors)

	def __getitem__(self, idx):
		anchor = self.anchors[idx]
		pos = self.positives[idx]
		return anchor, pos

