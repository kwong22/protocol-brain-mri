#!/usr/bin/python

import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import *
import model.lstm_lm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config_path):
	with open(config_path, 'r') as f:
		config = json.load(f)

	# Load training dataset
	train_dataset = get_train_dataset(config['train_data_path'])

	# Data Preprocessing

	# Load dictionary of out-of-vocabulary tokens and their replacements
	misspelled = {}
	if config['out_of_vocab_path'] is not None:
		misspelled = load_misspelled(config['out_of_vocab_path'])

	# Build vocabulary with replacements as specified
	vocab = build_vocab_with_corrections(train_dataset, my_tokenizer, misspelled)

	# Load BioWordVec embeddings into matching tokens of vocabulary
	print('Loading embeddings...')
	load_biowordvec_embeddings(config['biowordvec_path'], vocab)
	print('Done')


	# Text pipeline returns a list of tokens
	text_pipeline = lambda x: [vocab[token] for token in tokenize_with_corrections(x, my_tokenizer, misspelled)]

	def lm_collate_batch(batch):
	    """Collates batch for language model. In particular, converts tokens to
	        indices within vocabulary and adds padding to end of sequences as
	        necessary to make all sequences in the batch the same length.

	    Args:
	        batch: list of tuples (text, label), length of list is the batch size

	    Returns:
	        tuple of two lists:
	            lists of tokenized texts (strings replaced by indices into vocab)
	            lists of labels (next words, strings replaced by indices into vocab)
	    """
	    # Language model batch
	    # text_list: (batch size, sequence length)
	    # label_list: (batch size, sequence length) - since predicting next tokens
	    text_list = []
	    label_list = []
	    len_list = []
	    for text, _ in batch: # receives a batch of (text, label)
	        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
	        if len(processed_text) > 1:
	            text_list.append(processed_text[:-1])
	            label_list.append(processed_text[1:])
	            len_list.append(len(processed_text) - 1)
	    
	    # pack_padded_sequence expects lens sorted in decreasing order
	    len_order = np.flip(np.argsort(len_list)) # sorted in ascending, then flip
	    text_list = [text_list[i] for i in len_order]
	    label_list = [label_list[i] for i in len_order]
	    len_list = [len_list[i] for i in len_order]
	    
	    pad_idx = 1 # vocab.stoi['<pad>'] = 1
	    
	    text_list = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
	    label_list = pad_sequence(label_list, batch_first=True, padding_value=pad_idx)
	    len_list = torch.tensor(len_list, dtype=torch.int64)
	    
	    return text_list.to(device), label_list.to(device)

	# Language Model
	vocab_size = len(vocab)
	embed_size = vocab.vectors.shape[1] # matches pre-trained embeddings
	hidden_size = config['hidden_size']
	num_layers = config['num_layers']
	dropout = config['dropout']
	freeze_embeds = True

	lm_model = model.lstm_lm.LSTMLanguageModel(
		vocab_size,
		embed_size,
		hidden_size,
		num_layers,
		dropout,
		None,
		freeze_embeds,
	).to(device)


	# Initialize optimizer for language model
	lm_optimizer = torch.optim.Adam(lm_model.parameters(), lr=config['learning_rate'])

	# Split dataset randomly into training and validation sets
	num_train = int(len(train_dataset) * 0.8)
	split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

	lm_train_dataloader = DataLoader(split_train_, batch_size=config['batch_size'],
	                                 shuffle=True, collate_fn=lm_collate_batch)
	lm_valid_dataloader = DataLoader(split_valid_, batch_size=config['batch_size'],
	                                 shuffle=True, collate_fn=lm_collate_batch)

	# Train language model
	train(
		lm_model,
		lm_train_dataloader,
		lm_valid_dataloader,
		model.lstm_lm.loss_fn,
		lm_optimizer,
		model.lstm_lm.accuracy,
		weight=None,
		num_epochs=config['lm_epochs'],
	)

	# Save weights
	save_checkpoint(config['lm_weights_path'], lm_model, lm_optimizer)


if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1])
	else:
		print('Program expects: python train_lm.py <config_path>, where config_path is the path to a config in json format.')