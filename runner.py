import torch
from tqdm import tqdm
import sacrebleu

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
			'best_bleu_test': self.best_bleu_test
		}
		torch.save(checkpoint, name)

	def load_model(self, model, name, params, load_opt):
		checkpoint = torch.load(name)
		model.load_state_dict(checkpoint['model'])
		
		if load_opt:
			self.optimizer.load_state_dict(checkpoint['optimizer'])

		return checkpoint['epoch'], checkpoint['best_bleu_test']

	def fit_one_epoch(self, model, params, epoch):
		model.train()
		train_loss = 0.0
		for i, batch in enumerate(tqdm(self.train_dl, desc = f'Epoch {epoch}')):
			self.optimizer.zero_grad() # TODO: Should we clip gradient?
			output = model(**batch)
			loss = output[0] # self.loss_fn(output['loss'], output['target'])
			loss.backward()
			self.optimizer.step()
			self.cycle_scheduler.step()
			train_loss = loss.item()
			# translated_sents.extend(output['translated_src'])
			# target_sents.extend(batch['target_sent'])
		# bleu_score = sacrebleu.corpus_bleu(translated_sents, target_sentences).score
		print(f'Epoch {epoch}:')
		print(f'\tLoss: {loss}')

	def test(self, model, params, epoch):
		pass

	def train(self, model, params):
		self.best_train_loss = float('Inf')
		self.best_bleu_test = float('Inf')
		model = model.cuda()
		self.optimizer = optim.Adam(model.parameters(), lr = lr)
		last_epoch = 0
		last_batch = -1
		steps_per_epoch = len(self.train_dl)

		if params.load_model:
			last_epoch, self.best_bleu_test = self.load_model(model, params, load_opt = not params.test)
			last_batch = (last_epoch - 1) * steps_per_epoch

		if params.test:
			self.test(model, params, last_epoch)
			return
		
		self.cycle_scheduler = optim.lr_scheduler.OneCycleLR(optimizer = self.optimizer, max_lr = params.lr, 
			epochs = params.epochs, steps_per_epoch = steps_per_epoch, div_factor = 10, final_div_factor = 1e4, 
			last_epoch = last_batch, pct_start  =  0.2, anneal_strategy = 'linear')

		for epoch in params.epochs:
			self.fit_one_epoch(model, params, epoch)
			# self.test(model, params)

