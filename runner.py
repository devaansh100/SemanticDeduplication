import torch
from tqdm.autonotebook import tqdm
import torch.optim as optim
import os

class Runner:
	def __init__(self, train_dl, test_dl, loss_fn):
		self.train_dl = train_dl
		self.test_dl = test_dl
		self.loss_fn = loss_fn

	def save_model(self, model, name, epoch):
		checkpoint = {
			'model': model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'epoch': epoch,
		}
		torch.save(checkpoint, name)

	def load_model(self, model, name, params, load_opt):
		checkpoint = torch.load(name)
		model.load_state_dict(checkpoint['model'])
		
		if load_opt:
			self.optimizer.load_state_dict(checkpoint['optimizer'])

		return checkpoint['epoch']

	def fit_one_epoch(self, model, params, epoch):
		model.train()
		train_loss = 0.0
		pbar = tqdm(self.train_dl, desc = f'Epoch: {epoch}, Loss: 0.0')
		for i, batch in enumerate(pbar):
			for key in batch:
				batch[key] = batch[key].cuda()
			self.optimizer.zero_grad()
			output = model(**batch)
			loss = output[0]
			loss.backward()
			self.optimizer.step()
			self.cycle_scheduler.step()
			train_loss += loss.item()
			pbar.set_postfix(loss = train_loss/(i+1))
		
		if self.best_train_loss < loss:
			self.best_train_loss = loss
			self.save_model(model, f'{params.model_name}/model_best_loss.pth', epoch)


	def test_similarity(self, model, params, epoch):
		model.eval()
		for i, batch in enumerate(tqdm(self.test_dl, desc = f'Epoch {epoch}')):
			for key in batch:
				batch[key] = batch[key].cuda()
			output = model.forward_similarity(**batch)
			for article_1, article_2, score in zip(batch['article_1_ids'], batch['article_2_ids'], output):
				print(f'\nArticle 1: {model.tokenizer.decode(article_1, skip_special_tokens=True, clean_up_tokenization_spaces=True)}')
				print(f'Article 2: {model.tokenizer.decode(article_2, skip_special_tokens=True, clean_up_tokenization_spaces=True)}')
				print(f'Similarity: {score}\n')

	def test_generation(self, model, params, epoch):
		model.eval()
		for i, batch in enumerate(tqdm(self.train_dl, desc = f'Epoch {epoch}', total = 1)):
			for key in batch:
				batch[key] = batch[key].cuda()
			output = model(**batch, mode = 'generate')
			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output]
			target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in batch['target_ids']]
			for pred, target in zip(preds, target):
				print(f'\nGenerated Article: {pred}')
				print(f'Target Article: {target}\n')
			break

	def train(self, model, params):
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		self.best_train_loss = float('Inf')
		model = model.cuda()
		self.optimizer = optim.Adam(model.parameters(), lr = params.lr)
		last_epoch = 0
		last_batch = -1
		steps_per_epoch = len(self.train_dl)

		if params.load_model:
			last_epoch = self.load_model(model, params, load_opt = not params.test)
			last_batch = (last_epoch - 1) * steps_per_epoch

		if params.test:
			self.test_similarity(model, params, last_epoch)
			return
		
		self.cycle_scheduler = optim.lr_scheduler.OneCycleLR(optimizer = self.optimizer, max_lr = params.lr, 
			epochs = params.epochs, steps_per_epoch = steps_per_epoch, div_factor = 10, final_div_factor = 1e4, 
			last_epoch = last_batch, pct_start  =  0.2, anneal_strategy = 'linear')
		
		print('Zero-Shot testing of article similarity with pretrained T5 weights')
		self.test_similarity(model, params, 0)
		print('Zero-Shot testing of article generation with pretrained T5 weights')
		self.test_generation(model, params, epoch)
		for epoch in range(params.epochs):
			self.fit_one_epoch(model, params, epoch)
			self.test_similarity(model, params, epoch)
			self.test_generation(model, params, epoch)

