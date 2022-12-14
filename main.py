import torch
import argparse
from utils import *
from dataset import *
from runner import *
from torch.utils.data import DataLoader
import torch.nn as nn
from model import *
import random
import numpy as np
import torch
import os

def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

collate_util = lambda key, i, b : torch.cat([x[i][key] for x in b])
def collate_contrastive(batch):
	out = {'positives_encoded': {}, 'anchor_encoded': {}}
	out['anchor_encoded']['input_ids']      = collate_util('input_ids', 0, batch)
	out['anchor_encoded']['attention_mask'] = collate_util('attention_mask', 0, batch)
	# out['anchor_encoded']['token_type_ids'] = torch.zeros_like(out['anchor_encoded']['attention_mask'])

	out['positives_encoded']['input_ids']      = collate_util('input_ids', 1, batch)
	out['positives_encoded']['attention_mask'] = collate_util('attention_mask', 1, batch)
	# out['positives_encoded']['token_type_ids'] = torch.zeros_like(out['positives_encoded']['attention_mask'])
	return out

def collate(batch):
	entities_ids      = collate_util('input_ids', 0, batch)
	entities_att_mask = collate_util('attention_mask', 0, batch)
	articles_ids      = collate_util('input_ids', 1, batch)
	return {'source_ids': entities_ids, 'source_mask': entities_att_mask, 'target_ids': articles_ids[:, :-1], 'lm_labels': articles_ids[:, 1:]}

def collate_test(batch):
	article_1_ids      = collate_util('input_ids', 0, batch)
	article_1_att_mask = collate_util('attention_mask', 0, batch)
	article_2_ids      = collate_util('input_ids', 1, batch)
	article_2_att_mask = collate_util('attention_mask', 1, batch)
	labels 			   = torch.tensor([x[2] for x in batch])
	return {'article_1_ids': article_1_ids, 'article_1_att_mask': article_1_att_mask, 'article_2_ids': article_2_ids, 'article_2_att_mask': article_2_att_mask, 'labels': labels}

def main(params):
	init_seed(42)
	entities, files = get_conll_data(f'{params.data_dir}/Final_CONLL')
	articles = get_raw_data(f'{params.data_dir}/Final_TEXT', files)
	if params.train_contrastive:
		positives = get_raw_data(f'{params.data_dir}/Final_POS', files)

	test_labels, test_articles_1,  test_articles_2 = read_test_articles(f'{params.data_dir}/Test_files')
	model = PositiveGenerator() if not params.train_contrastive else ContrastiveModel(params.positive_method)
	print('Tokenizing Articles')
	tokenized_articles = tokenize(articles, model.tokenizer)
	print('Tokenizing Entities')
	tokenized_entities = tokenize(entities, model.tokenizer)
	print('Tokenizing Test articles')
	tokenized_test_1 = tokenize(test_articles_1, model.tokenizer)
	tokenized_test_2 = tokenize(test_articles_2, model.tokenizer)
	if params.train_contrastive:
		print('Tokenizing Positive Articles')
		tokenized_positives = tokenize(positives, model.tokenizer)

	if not params.train_contrastive:
		train_ds = WildlifeDataset(tokenized_entities, tokenized_articles, model.tokenizer, params)
	else:
		train_ds = ContrastiveDataset(tokenized_articles, tokenized_positives, model.tokenizer, params)
	test_ds = WildlifeDataset(tokenized_test_1, tokenized_test_2, model.tokenizer, params, labels)

	train_dl = DataLoader(train_ds, batch_size = params.batch_size, shuffle = not params.test, pin_memory = True, num_workers = 4, collate_fn = collate if not params.train_contrastive else collate_contrastive)
	test_dl = DataLoader(test_ds, batch_size = params.batch_size, pin_memory = True, num_workers = 4, collate_fn = collate_test if not params.train_contrastive else collate_contrastive)
	runner = RunnerContrastive(train_dl, test_dl) if params.train_contrastive else Runner(train_dl, test_dl, files)
	runner.train(model, params)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--job', dest = 'job_name', type = str, default = '', help = 'Name of the job')
	parser.add_argument('--lm', dest = 'load_model', type = str, default = '', help = 'Name of model to be loaded')
	parser.add_argument('--test', action = 'store_true')
	parser.add_argument('--train_contrastive', action = 'store_true')
	parser.add_argument('--mb', dest = 'batch_size', type = int, default = 128)
	parser.add_argument('--lr', type = float, default = 1e-5)
	parser.add_argument('--subset', type = int, default = 5)
	parser.add_argument('--data_dir', type = str, default = 'Annotated_Articles')
	parser.add_argument('--epochs', type = int, default = 15)
	parser.add_argument('--seq_len', type = int, default = 512)
	parser.add_argument('--positive_method', type = str, choices = ['generate', 'masking'], default = 'masking')
	params = parser.parse_args()
	main(params)
