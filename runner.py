import torch
from tqdm.autonotebook import tqdm
import torch.optim as optim
import os

class Runner:
	def __init__(self, train_dl, test_dl, files = None):
		self.train_dl = train_dl
		self.test_dl = test_dl
		self.files = files

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
		pbar = tqdm(self.train_dl, desc = f'Train Epoch: {epoch}')
		for i, batch in enumerate(pbar):
			batch = self.send_to_cuda(batch)
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

	def send_to_cuda(self, batch):
		for key in batch:
			batch[key] = batch[key].cuda()
		return batch

	def test(self, model, params, epoch):
		model.eval()
		self.articles_gen = []
		self.train_dl.shuffle = False
		for i, batch in enumerate(tqdm(self.train_dl, desc = f'Generation Epoch {epoch}', total = 1)): # Generates the positive articels from the entities
			batch = self.send_to_cuda(batch)
			output = model(**batch, mode = 'generate')
			preds = model.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
			target = model.tokenizer.batch_decode(batch['target_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
			self.articles_gen.extend(preds)
			# for pred, target in zip(preds, target):
			# 	print(f'\nGenerated Article: {pred}')
			# 	print(f'Target Article: {target}\n')

		os.makedirs(f'{params.data_dir}/Final_POS', exist_ok = True)
		for i, article in enumerate(self.articles_gen):
			with open(f'{params.data_dir}/Final_POS/{self.files[i].split("/")[-1].replace(".conll", ".txt")}', 'w') as f:
				f.write(article)

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
		
		self.cycle_scheduler = optim.lr_scheduler.OneCycleLR(optimizer = self.optimizer, max_lr = params.lr, 
			epochs = params.epochs, steps_per_epoch = steps_per_epoch, div_factor = 10, final_div_factor = 1e4, 
			last_epoch = last_batch, pct_start  =  0.2, anneal_strategy = 'linear')

		if params.test:
			self.test(model, params, last_epoch)
			return 0
		
		# print('Zero-Shot testing of article similarity with pretrained T5 weights')
		# self.test_similarity(model, params, 0)
		# print('Zero-Shot testing of article generation with pretrained T5 weights')
		# self.test_generation(model, params, 0)
		# for epoch in range(params.epochs):
		# 	self.fit_one_epoch(model, params, epoch + 1)
		# 	self.test_similarity(model, params, epoch + 1)
		# 	self.test_generation(model, params, epoch + 1)

class RunnerContrastive(Runner):
	def __init__(self, train_dl, test_dl):
		super().__init__(train_dl, test_dl)

	def send_to_cuda(self, batch):
		for loss_set in batch:
			if torch.is_tensor(loss_set):
				batch[loss_set] = batch[loss_set].cuda()
			else:
				for key in batch[loss_set]:
					batch[loss_set][key] = batch[loss_set][key].cuda()
		return batch

	def test(self, model, params, epoch):
		model.eval()
		correct_count = 0
		for i, batch in enumerate(tqdm(self.test_dl, desc = f'Epoch {epoch}')):
			batch = self.send_to_cuda(batch)
			labels = batch.pop('labels')
			output = model.forward(**batch)
			preds = torch.where(output > 0.95, 1, 0)
			correct_count += (labels == preds).sum().item()
			# for article_1, article_2, score in zip(batch['anchor_encoded']['input_ids'], batch['positives_encoded']['input_ids'], output):
			# 	print(f'\nArticle 1: {model.tokenizer.decode(article_1, skip_special_tokens=True, clean_up_tokenization_spaces=True)}')
			# 	print(f'Article 2: {model.tokenizer.decode(article_2, skip_special_tokens=True, clean_up_tokenization_spaces=True)}')
			# 	print(f'Similarity: {score}\n')
		print(f'Epoch {epoch}: Test Accuracy = {round(100*correct_count/len(self.test_dl.dataset), 2)}%')

	def train(self, model, params):
		ret_val = super().train(model, params)
		if ret_val == 0:
			return

		# self.test(model, params, 0)
		for epoch in range(params.epochs):
			self.fit_one_epoch(model, params, epoch + 1)
			self.test(model, params, epoch + 1)







