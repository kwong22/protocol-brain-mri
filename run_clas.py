#!/usr/bin/python

import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import *
from metrics import *
import model.lstm_clas

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config_path, text):
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


	# Load saved weights
	load_checkpoint(config['clas_weights_path'], clas_model)

	# Process text into acceptable input for the model
	processed_text = torch.unsqueeze(torch.tensor(text_pipeline(text)), dim=0).to(device)

	# Pass processed text into the model
	clas_model.eval()
	with torch.no_grad():
		output = clas_model(processed_text)
		pred = target_names[torch.argmax(output)]
		print(f'Input: {text}\nPrediction: {pred}')


if __name__ == '__main__':
	if len(sys.argv) == 3:
		main(sys.argv[1], sys.argv[2])
	else:
		print('Program expects: python run_clas.py <config_path> <text>, where config_path is the path to a config in json format and text is a string to classify.')