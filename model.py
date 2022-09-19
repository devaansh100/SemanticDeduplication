import torch
import torch.nn as nn
from transformers import T5TokenizerFast, T5ForConditionalGeneration

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
		self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
		self.seq_len = 512

	def forward(self, source_mask, source_ids, target_ids, lm_labels, mode = 'train'):
		# TODO: Add attention masks etc etc here.
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

		elif mode == 'get_similarity':
			pass
			