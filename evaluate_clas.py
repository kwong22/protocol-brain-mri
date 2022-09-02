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


	# Load saved weights
	load_checkpoint(config['clas_weights_path'], clas_model)

	test_dataset_actual, test_dataset_phys = get_test_datasets(config['test_data_path'])
	# test_dataset_actual - labels are originally assigned protocols
	# test_dataset_phys - labels were determined by physician consensus

	test_dataloader_actual = DataLoader(test_dataset_actual,
	                                    batch_size=config['batch_size'],
	                                    shuffle=False,
	                                    collate_fn=clas_collate_batch)

	test_dataloader_phys = DataLoader(test_dataset_phys,
	                                  batch_size=config['batch_size'],
	                                  shuffle=False,
	                                  collate_fn=clas_collate_batch)

	# Evaluate on originally assigned protocols
	_, actual_accu = evaluate(clas_model, test_dataloader_actual, model.lstm_clas.loss_fn, model.lstm_clas.accuracy)
	print('Accuracy on originally assigned protocols: {:8.6f}'.format(actual_accu))

	# Evaluate on physician consensus labels
	_, phys_accu = evaluate(clas_model, test_dataloader_phys, model.lstm_clas.loss_fn, model.lstm_clas.accuracy)
	print('Accuracy on physician consensus labels: {:8.6f}'.format(phys_accu))


	# Evaluate model as a Clinical Decision Support tool

	# Hyperparameter tuning

	# thresh - confidence level threshold demarcating automatic and CDS modes
	thresh_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

	# k - number of protocols to suggest in CDS mode
	k_list = [1, 2, 3, 4, 5]

	stats = []

	print('\nEvaluating as a CDS tool...')
	print('Format: thresh, k : ppv/sens/f1 ppv_auto/sens_auto(% auto) ppv_cds/sens_cds(% cds)')

	for t in thresh_list:
	    for k in k_list:
	        cm_auto, cm_cds = get_confusion_matrix(
	            clas_model,
	            test_dataloader_phys, # Labels from physican consensus
	            target_names,
	            thresh=t,
	            k=k,
	            return_cds=True,
	        )
	        ppv, sens, f1 = calculate_metrics_from_cm(cm_auto + cm_cds, target_names, print_metrics=False)
	        ppv_auto, sens_auto, _ = calculate_metrics_from_cm(cm_auto, target_names, print_metrics=False)
	        ppv_cds, sens_cds, _ = calculate_metrics_from_cm(cm_cds, target_names, print_metrics=False)
	        
	        stat = [t, k, ppv, sens, f1, ppv_auto, sens_auto, np.sum(cm_auto) / np.sum(cm_auto + cm_cds) * 100,
	        ppv_cds, sens_cds, np.sum(cm_cds) / np.sum(cm_auto + cm_cds) * 100]
	        stats.append(stat)
	        
	        print('thresh={:03.1f}, k={} : {:.3f}/{:.3f}/{:.3f} {:.3f}/{:.3f} ({:04.1f}%) {:.3f}/{:.3f} ({:04.1f}%)'.format(*stat))


if __name__ == '__main__':
	if len(sys.argv) == 2:
		main(sys.argv[1])
	else:
		print('Program expects: python evaluate_clas.py <config_path>, where config_path is the path to a config in json format.')