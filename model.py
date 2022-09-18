import torch
import torch.nn as nn
from transformers import T5TokenizerFast, T5ForConditionalGeneration

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.tokenizer = T5TokenizerFast.from_pretrained("t5-small")
		self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

	def forward(self, source_mask, source_ids, target_ids, lm_labels):
		# TODO: Add attention masks etc etc here.
		outputs = self.t5(
			input_ids = source_ids,
			attention_mask = source_mask, 
			decoder_input_ids = target_ids,
			labels = lm_labels
			)
		return outputs