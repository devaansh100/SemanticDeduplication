import torch
import torch.nn as nn
from transformers import T5TokenizerFast, T5ForConditionalGeneration

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
		self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
		self.seq_len = 512
		self.sim = nn.CosineSimilarity(dim = 0)

	def get_scores(self, article_1_emb, article_2_emb, article_1_att_mask, article_2_att_mask):
		score = torch.zeros(article_1_emb.shape[0])
		for b in range(article_1_emb.shape[0]):
			n_article_1 = torch.sum(article_1_att_mask[b])
			n_article_2 = torch.sum(article_2_att_mask[b])
			sim_matrix = torch.zeros((n_article_1, n_article_2))
			for i in range(n_article_1):
				for j in range(n_article_2):
					sim_matrix[i, j] = self.sim(article_1_emb[b, i], article_2_emb[b, j])
			score[b] = torch.sum(torch.max(sim_matrix, dim = 1)[0]) + torch.sum(torch.max(sim_matrix, dim = 0)[0])
			score[b] = score[b]/(n_article_1 + n_article_2)
		return score

	def forward(self, source_mask, source_ids, target_ids, lm_labels, mode = 'train'):
		if mode == 'train':
			outputs = self.t5(
				input_ids = source_ids,
				attention_mask = source_mask, 
				decoder_input_ids = target_ids,
				labels = lm_labels
				)
			return outputs
		elif mode == 'generate':
			generated_ids = self.t5.generate(
								input_ids = source_ids,
								attention_mask = source_mask, 
								max_length = self.seq_len, 
								num_beams = 5,
								early_stopping = True
							)
		return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

	def forward_similarity(self, article_1_ids, article_1_att_mask, article_2_ids, article_2_att_mask, article_1_txt = None, article_2_txt = None):
		article_1_emb = self.t5.encoder(
				input_ids = article_1_ids,
				attention_mask = article_1_att_mask, 
			).last_hidden_state
		article_2_emb = self.t5.encoder(
				input_ids = article_2_ids,
				attention_mask = article_2_att_mask, 
			).last_hidden_state

		score = self.get_scores(article_1_emb, article_2_emb, article_1_att_mask, article_2_att_mask)
		return score