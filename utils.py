import os
import re
import string
import time
from tqdm import tqdm

from collections import Counter
import numpy as np
import pandas as pd
import torch
from torchtext.vocab import Vocab

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import KeyedVectors

def get_train_dataset(path):
    """Loads dataset for training and validation sets.

    Args:
        path (string): path to CSV file to read from; has columns 'ID', 'label',
            'text'

            ID is a unique number assigned to the exam
            label is a number indicating the originally assigned protocol
            text is the text information associated with the exam

    Returns:
        list of tuples (text, label)
    """
    df = pd.read_csv(path)
    df.columns = ['ID', 'label', 'text'] # rename columns for DataLoader
    return [(df['text'][idx], df['label'][idx]) for idx in range(len(df['label']))]

def get_test_datasets(path):
    """Loads dataset for test set.

    Args:
        path (string): path to CSV file to read from; has columns 'ID',
            'reviewer1', 'reviewer2', 'label', 'truth', 'text'

            ID is a unique number assigned to the exam
            reviewer1 is the label assigned to the exam by the first reviewer
            reviewer2 is the label assigned to the exam by the second reviewer
            label is a number indicating the originally assigned protocol
            truth is a number indicating the label determined by physician consensus
            text is the text information associated with the exam

    Returns:
        tuple of two lists:
            First list contains tuples (text, label) where label is the
                originally assigned protocol
            Second list contains tuples (text, truth) where truth is the
                protocol determined by physician consensus
    """
    df = pd.read_csv(path)
    df.columns = ['ID', 'reviewer1', 'reviewer2', 'label', 'truth', 'text']
    
    ds_label = [(df['text'][idx], df['label'][idx]) for idx in range(len(df['text']))]
    ds_truth = [(df['text'][idx], df['truth'][idx]) for idx in range(len(df['text']))]
    
    return ds_label, ds_truth


def normalize_text(s):
    """
    Converts text to lowercase, replaces certain terms, removes ICD codes,
    removes punctuation, removes numbers, and removes extra whitespace.

    Args:
        s (string): text to be normalized

    Returns:
        normalized text
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_icd(text):
        # Remove ICD9 codes (brackets and everything in between them)
        return re.sub(r'\[.*?\]', '', text)
    
    def remove_punc(text):
        punc_no_space = set('\'') # replace with nothing
        punc_space = set('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~') # replace with space
        # keep hyphens
                
        out = []
        
        for c in text:
            if c in punc_no_space:
                continue # replace with nothing
            elif c in punc_space:
                out.append(' ') # replace with space
            else:
                out.append(c) # keep current character
        
        return ''.join(out)
    
    def remove_numbers(text):
        return re.sub(r'\d+', '', text)
        
    def replace_known(text):
        known_words = {
            'r/o': 'ruleout',
            'y/o': 'yearold',
            'yo': 'yearold',
            'y.o.': 'yearold',
            's/p': 'statuspost',
            'h/o': 'historyof',
            'hx': 'history',
            'n/v': 'nausea vomiting',
            'f/u': 'followup',
            'p/w': 'presentedwith',
            'sx': 'symptom',
            'w/u': 'workup',
        }
        
        tokens = text.split()
        
        out = []
        
        for token in tokens:
            if token in known_words:
                out.append(known_words[token]) # replace matching token
            else:
                out.append(token) # keep current token
        
        return ' '.join(out)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_numbers(remove_punc(remove_icd(replace_known(lower(s))))))


def my_tokenizer(s):
    """Normalizes and splits text into tokens.

    Args:
        s (string): text to be tokenized

    Returns:
        list of normalized tokens
    """
    return normalize_text(s).split(' ')


def build_vocab(dataset, tokenizer, vectors=None):
    """Builds vocabulary from all text information in a dataset.

    Args:
        dataset: list of tuples (text, label)
        tokenizer: function that splits text into tokens
        vectors: pre-trained vectors (optional)
            specified in https://pytorch.org/text/0.9.0/vocab.html

    Returns:
        a torchtext.vocab.Vocab object
    """
    counter = Counter()

    for text, label in dataset:
        counter.update(tokenizer(text))

    return Vocab(counter, min_freq=1, vectors=vectors)


def load_misspelled(path):
    """Loads replacement rules for out-of-vocabulary tokens.

    Args:
        path (string): path to file containing replacement rules with the form
            `count misspelled-token -> [replacement-token]* replacement-choice`

    Returns:
        dictionary where key is misspelled token, and value is its replacement
    """
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(line.split())

    misspelled = {}
    
    # Exclude first two lines, which indicate counts
    for line in lines[2:]:
        value = ''
        # check if a numeric replacement option was specified
        if line[-1].isnumeric():
            offset = int(line[-1])
            if offset > 0:
                # change to one of the other options
                temp = line[line.index('->') + offset]
                # '~' for space
                value = temp.replace('~', ' ')
            else:
                # keep same term
                value = line[1]
        else:
            if line[-1] == '#':
                # '#' for deletion
                value = ''
            else:
                # choose first option
                if len(line) > line.index('->') + 1:
                    temp = line[line.index('->') + 1]
                    # '~' for space
                    value = temp.replace('~', ' ')
                else:
                    # keep same term
                    value = line[1]
        
        misspelled[line[1]] = value
    
    return misspelled


def tokenize_with_corrections(s, tokenizer, misspelled={}):
    """Converts text into tokens, with certain tokens replaced as specified.

    Args:
        s (string): text to tokenize
        tokenizer: function that splits text into a list of tokens
        misspelled: dictionary of out-of-vocabulary tokens and their replacements

    Returns:
        list of revised tokens
    """
    tokens = tokenizer(s)
        
    # Replace misspelled tokens
    for i in range(len(tokens)):
        if tokens[i] in misspelled:
            tokens[i] = misspelled[tokens[i]]

    # Resolve whitespace
    tokens = ' '.join(tokens).split()
    
    return tokens


def build_vocab_with_corrections(dataset, tokenizer, misspelled={}, vectors=None):
    """Builds vocabulary from all text information in a dataset, with certain
        tokens replaced as specified.

    Args:
        dataset: list of tuples (text, label)
        tokenizer: function that splits text into tokens
        misspelled: dictionary of out-of-vocabulary tokens and their replacements
        vectors: pre-trained vectors (optional)
            specified in https://pytorch.org/text/0.9.0/vocab.html

    Returns:
        a torchtext.vocab.Vocab object
    """
    counter = Counter()

    for text, label in dataset:
        counter.update(tokenize_with_corrections(text, tokenizer, misspelled))

    return Vocab(counter, min_freq=1, vectors=vectors)


def load_biowordvec_embeddings(path, vocab):
    """Loads BioWordVec embeddings into corresponding tokens of a Vocab object.

    Args:
        path (string): path to BioWordVec embeddings
        vocab (torchtext.vocab.Vocab): vocabulary
    """
    embed_dim = 200 # set by BioWordVec
    vocab_size = len(vocab.itos)
    
    embeds = np.zeros((vocab_size, embed_dim))
    
    vectors = KeyedVectors.load_word2vec_format(path, binary=True)
    
    for idx in range(vocab_size):
        word = vocab.itos[idx]
        
        if word in vectors:
            vector = vectors.get_vector(word)
            embeds[idx] = vector
    
    embeds = torch.from_numpy(embeds).float()
    vocab.set_vectors(vocab.stoi, embeds, embed_dim)


def save_checkpoint(checkpoint_path, model, optimizer=None):
    """Saves model and training parameters to specified path.

    Args:
        checkpoint_path (string): filename to save to
        model (torch.nn.Module): model containing parameters to save
        optimizer (torch.optim): optimizer containing parameters to save
    """
    state = {}
    state['model_state_dict'] = model.state_dict()
    if optimizer:
        state['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(state, checkpoint_path)

    print('model saved to {}'.format(checkpoint_path))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model parameters (state_dict) from specified path.

    Args:
        checkpoint_path (string): filename to load from
        model (torch.nn.Module): model for which parameters are loaded
        optimizer (torch.optim): optimizer for which parameters are loaded
    """
    if not os.path.exists(checkpoint_path):
        raise('File does not exist {}'.format(checkpoint_path))

    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    print('model loaded from {}'.format(checkpoint_path))


def load_encoder(checkpoint_path, model, last_layer_name):
    """Loads model parameters for all except last layer from specified path.

    Args:
        checkpoint_path (string): filename to load from
        model (torch.nn.Module): model for which parameters are loaded
        last_layer_name (string): name of layer to exclude
    """
    checkpoint = torch.load(checkpoint_path)

    # Extract states to load (all but last layer)
    states_to_load = {}
    for name, param in checkpoint['model_state_dict'].items():
        if not name.startswith(last_layer_name):
            states_to_load[name] = param

    # Copy current model state and update specified states
    model_state = model.state_dict()
    model_state.update(states_to_load)

    # Replace current model state
    model.load_state_dict(model_state)

    print('model loaded from {}'.format(checkpoint_path))


def compute_class_weights(dataset, target_names):
    """Compute class weights for unbalanced datasets. Algorithm from
        `sklearn.utils.class_weight.compute_class_weight`:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    Args:
        dataset: list of tuples (text, label)
        target_names: list of names of target labels (just needs correct length)

    Returns:
        Tensor of weights, one weight for each class
    """
    labels = [label for _, label in dataset]
    return torch.tensor(len(labels) / (len(target_names) * np.bincount(labels))).float()


def evaluate(model, dataloader, loss_fn, acc_fn, weight=None):
    """Calculates model loss and accuracy on a dataset.

    Args:
        model (torch.nn.Module): model to evaluate
        dataloader (torch.utils.data.DataLoader): data including text and label
        loss_fn: function that computes loss from predicted and actual labels
        acc_fn: function that computes accuracy from predicted and actual labels
        weight: weights to use with loss function (optional)

    Returns:
        tuple of average loss per example and average accuracy per batch
    """
    model.eval() # set to eval mode
    
    total_loss = 0
    total_acc = 0
    total_count = 0

    with torch.no_grad(): # do not compute gradients during evaluation
        for idx, (text, label) in enumerate(dataloader):
            pred_label = model(text)
            total_loss += loss_fn(pred_label, label, weight).item()
            total_acc += acc_fn(pred_label, label)
            total_count += label.shape[0] # adds batch size
    return total_loss/total_count, total_acc/len(dataloader)

def train(
    model,
    train_dataloader,
    valid_dataloader,
    loss_fn,
    optimizer,
    acc_fn,
    weight=None,
    num_epochs=1,
):
    """Trains model on a training set and, for each epoch, provides results
        on a validation set.

    Args:
        model (torch.nn.Module): model to train
        train_dataloader (torch.utils.data.DataLoader): training set
        valid_dataloader (torch.utils.data.DataLoader): validation set
        loss_fn: function that computes loss from predicted and actual labels
        optimizer (torch.optim): optimizer for training
        acc_fn: function that computes accuracy from predicted and actual labels
        weight: weights to use with loss function (optional)
        num_epochs: number of epochs of training set on which to train (default 1)
    """
    example_count = 0
    batch_count = 0
    
    pbar = tqdm() # monitor number of batches completed within epoch
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training
        model.train() # set to train mode

        total_train_loss = 0
        total_train_count = 0
        
        pbar.reset(total=len(train_dataloader)) # reset and reuse the bar
        pbar.set_description(desc='epoch {}/{}'.format(epoch, num_epochs))
        
        for idx, (text, label) in enumerate(train_dataloader):
            # Forward pass
            out = model(text)
            loss = loss_fn(out, label, weight) # calculate loss
            
            # Backward pass
            optimizer.zero_grad() # reset gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # clip gradients
            
            # Step with optimizer
            optimizer.step() # adjust parameters by gradients collected in backward pass
            
            example_count += len(label)
            batch_count += 1
            
            total_train_loss += loss.item()
            total_train_count += len(label) # adds batch size
            
            pbar.update() # increment bar by 1
        
        # Validation
        valid_loss, valid_accu = evaluate(model, valid_dataloader, loss_fn, acc_fn, weight)

        # Print progress for epoch
        tqdm.write('epoch {:3d} '
                   '| train_loss: {:8.6f} | valid_loss: {:8.6f} '
                   '| accuracy: {:8.6f} | time: {:5.2f}s'.format(epoch,
                                                                 total_train_loss / total_train_count,
                                                                 valid_loss,
                                                                 valid_accu,
                                                                 time.time() - start_time))
    
    pbar.close()

