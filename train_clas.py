#!/usr/bin/python

import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import *
import model.lstm_clas

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

	# Text pipeline returns a list of tokens
	text_pipeline = lambda x: [vocab[token] for token in tokenize_with_corrections(x, my_tokenizer, misspelled)]
	label_pipeline = lambda x: x

	# Fine tune for classification
	target_names = [
	    'MRI BRAIN WITHOUT/ WITH CONTRAST',
	    'MRI BRAIN FOR MS',
	    'MRI TUMOR',
	    'MRI BRAIN MENINGIOMA FOLLOW-UP',
	    'MRI BRAIN DEDICATED SEIZURE WITH CONTRAST',
	    'MRI STEALTH/ STRYKER/ MASK/ PRE-SURGICAL PLANNING',
	    'MRI FOR GAMMA KNIFE, BRAIN LAB, AND SRS',
	    'MRI CRANIAL NERVES 3-6 WITHOUT/ WITH CONTRAST',
	]

	# Calculate class weights for unbalanced dataset
	class_weights = compute_class_weights(train_dataset, target_names).to(device)

	def clas_collate_batch(batch):
	    """Collates batch for classifier. In particular, converts tokens to indices
	        within vocabulary and adds padding to end of sequences as
	        necessary to make all sequences in the batch the same length.

	    Args:
	        batch: list of tuples (text, label), length of list is the batch size

	    Returns:
	        tuple of two lists:
	            lists of tokenized texts (strings replaced by indices into vocab)
	            labels
	    """
	    # Classification batch
	    # text_list: (batch size, sequence length)
	    # label_list: (batch size)
	    text_list = []
	    label_list = []
	    len_list = []
	    for _text, _label in batch:
	        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
	        if len(processed_text) > 0:
	            text_list.append(processed_text)
	            len_list.append(len(processed_text))

	            label_list.append(label_pipeline(_label))
	    
	    # pack_padded_sequence expects lens sorted in decreasing order
	    len_order = np.flip(np.argsort(len_list)) # sorted in ascending, then flip
	    text_list = [text_list[i] for i in len_order]
	    label_list = [label_list[i] for i in len_order]
	    len_list = [len_list[i] for i in len_order]
	    
	    pad_idx = 1 # vocab.stoi['<pad>'] = 1
	    
	    text_list = pad_sequence(text_list, batch_first=True, padding_value=pad_idx)
	    label_list = torch.tensor(label_list, dtype=torch.int64)
	    len_list = torch.tensor(len_list, dtype=torch.int64)
	    
	    return text_list.to(device), label_list.to(device)


	# Classifier model

	# Use same parameters as language model
	vocab_size = len(vocab)
	embed_size = 200 # From BioWordVec embeddings
	hidden_size = config['hidden_size']
	num_layers = config['num_layers']
	dropout = config['dropout']
	num_classes = len(target_names)

	clas_model = model.lstm_clas.LSTMClassifier(
		vocab_size,
		embed_size,
		hidden_size,
		num_layers,
		dropout,
		num_classes,
		None,
	).to(device)


	# Load all but last layer (encoder)
	load_encoder(config['lm_weights_path'], clas_model, last_layer_name='hidden2out')

	# Initialize optimizer for classifier
	clas_optimizer = torch.optim.Adam(clas_model.parameters(), lr=config['learning_rate'])

	# Split dataset randomly into training and validation sets
	num_train = int(len(train_dataset) * 0.8)
	split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

	clas_train_dataloader = DataLoader(split_train_, batch_size=config['batch_size'],
	                                   shuffle=True, collate_fn=clas_collate_batch)
	clas_valid_dataloader = DataLoader(split_valid_, batch_size=config['batch_size'],
	                                   shuffle=True, collate_fn=clas_collate_batch)

	# Train classifier
	train(
		clas_model,
		clas_train_dataloader,
		clas_valid_dataloader,
		model.lstm_clas.loss_fn,
		clas_optimizer,
		model.lstm_clas.accuracy,
		weight=class_weights,
		num_epochs=config['clas_epochs'],
	)

	# Save weights
	save_checkpoint(config['clas_weights_path'], clas_model, clas_optimizer)


if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1])
	else:
		print('Program expects: python train_clas.py <config_path>, where config_path is the path to a config in json format.')