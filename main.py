import torch
import argparse
from utils import *
from dataset import *
from runner import *
from torch.utils.data import DataLoader
import torch.nn as nn
from model import *

def collate(batch):

	entities_ids      = torch.cat([x[0]['input_ids'] for x in batch])
	entities_att_mask = torch.cat([x[0]['attention_mask'] for x in batch])
	articles_ids      = torch.cat([x[1]['input_ids'] for x in batch])
	return {'source_ids': entities_ids, 'source_mask': entities_att_mask, 'target_ids': articles_ids[:-1], 'lm_labels': articles_ids[1:]}

def main(params):
	entities, files = get_conll_data(f'{params.data_dir}/Final_CONLL')
	articles = get_raw_data(f'{params.data_dir}/Final_TEXT', files)
	
	model = Model()
	print('Tokenizing Articles')
	tokenized_articles = tokenize(articles, model.tokenizer)
	print('Tokenizing Entities')
	tokenized_entities = tokenize(entities, model.tokenizer)
	train_ds = WildlifeDataset(tokenized_entities, tokenized_articles, model.tokenizer, params)
	test_ds = WildlifeDataset(tokenized_entities, tokenized_articles, model.tokenizer, params)

	train_dl = DataLoader(train_ds, batch_size = params.batch_size, shuffle = True, pin_memory = True, num_workers = 6, collate_fn = collate)
	test_dl = DataLoader(test_ds, batch_size = params.batch_size, pin_memory = True, num_workers = 6, collate_fn = collate)

	runner = Runner(train_dl, test_dl, nn.NLLLoss())
	runner.train(model, params)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--job', dest = 'job_name', type = str, default = '', help = 'Name of the job')
	parser.add_argument('--lm', dest = 'load_model', type = str, default = '', help = 'Name of model to be loaded')
	parser.add_argument('--test', action = 'store_true')
	parser.add_argument('--mb', dest = 'batch_size', type = int, default = 128)
	parser.add_argument('--lr', type = float, default = 5e-3)
	parser.add_argument('--subset', type = int, default = 5)
	parser.add_argument('--data_dir', type = str, default = 'Re_Annotated_Articles')
	parser.add_argument('--epochs', type = int, default = 15)
	params = parser.parse_args()
	main(params)