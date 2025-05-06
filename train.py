#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import json
import time
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk
import pickle

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import PreTrainedTokenizerFast
from sklearn.model_selection import train_test_split


import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import ast
import warnings
import json
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import os



# Load T5 model and tokenizer
model_name = "t5-large"



dataset_dict = DatasetDict()
# Load the dataset from disk
dataset_dict['train'] = load_from_disk('data/cs_paper_author_over2pub/dataset_directory/train')
dataset_dict['test'] = load_from_disk('data/cs_paper_author_over2pub/dataset_directory/test')
train_data = dataset_dict['train']
test_data = dataset_dict['test']




# # Define Tokenizers

# ## Task tokenizer
task_tokenizer = T5Tokenizer.from_pretrained(model_name)
print(task_tokenizer)


# ## Team tokenizer
cs_paper_author = pd.read_csv("./output/unique_author_published_2plus_cs_paper.csv")
list_cs_paper_author = cs_paper_author["author_id"].unique()
print(len(list_cs_paper_author))


class SimpleTeamTokenizer:
    def __init__(self, team_ids):
        self.team_ids = team_ids
        self.token_to_id = {id_: idx for idx, id_ in enumerate(team_ids)}
        self.token_to_id['<pad>'] = len(team_ids)  # Define the padding token
        self.id_to_token = {idx: id_ for idx, id_ in enumerate(team_ids)}
        self.id_to_token[len(team_ids)] = '<pad>'  # Map the padding token to its index

    def string_to_list(self, text):
        """
        Tokenize a string representation of a list into individual tokens (IDs).
        If the text is not a valid list format, return an empty list.
        """
        try:
            # Safely convert string representation of a list to an actual list of tokens
            token_list = ast.literal_eval(text)
            if isinstance(token_list, list):
                return token_list  # Return the parsed list of tokens
        except (ValueError, SyntaxError):
            return []  # Return an empty list if parsing fails

    def convert_tokens_to_ids(self, tokens):
        """
        Convert list of tokens (IDs as strings) to list of token IDs (as integers).
        If a token is not found, replace it with the ID for the padding token.
        """
        return [self.token_to_id.get(token, self.token_to_id['<pad>']) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Convert list of token IDs (integers) back to the original token strings (IDs).
        If skip_special_tokens is True, exclude padding tokens from the output.
        """
        tokens = [self.id_to_token.get(id_, '<pad>') for id_ in ids]
        if skip_special_tokens:
            # Filter out the <pad> tokens
            tokens = [token for token in tokens if token != '<pad>']
        return tokens

# Initialize the team tokenizer
team_tokenizer = SimpleTeamTokenizer(list_cs_paper_author)


# # Train

# ## Tokenize Train data


def tokenize_data(examples, task_tokenizer, team_tokenizer, max_input_length=512, max_target_length=17
                  ):
    # Tokenize task description (input)
    inputs = task_tokenizer(
        examples['input'], 
        max_length=max_input_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

    # Tokenize team author IDs (target)
    targets = [team_tokenizer.convert_tokens_to_ids(team_tokenizer.string_to_list(author_list)) 
               for author_list in examples['team']]
    
    # Pad or truncate the target author IDs to ensure uniform length
    max_len = max_target_length
    padded_targets = []
    for target in targets:
        if len(target) < max_len:
            target += [team_tokenizer.token_to_id['<pad>']] * (max_len - len(target))  # Pad to max length
        padded_targets.append(target[:max_len])  # Truncate if too long
        
    # Convert the padded targets to a tensor
    padded_targets_tensor = torch.tensor(padded_targets, dtype=torch.long)
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': padded_targets_tensor  # Labels should be a tensor of the padded targets
    }


def collate_fn(batch):
    # Convert everything in the batch to tensors and stack them
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
    labels = torch.stack([torch.tensor(x['labels']) for x in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }



# Tokenize training and validation data
train_tokenized = train_data.map(lambda x: tokenize_data(x, task_tokenizer, team_tokenizer), batched=True)

# DataLoader with collate_fn to handle variable-sized batches
train_dataloader = DataLoader(train_tokenized, batch_size=16, shuffle=True, collate_fn=collate_fn)


# ## Define StepLR and EarlyStopping

class StepLR:
    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        if self.last_epoch % self.step_size == 0 and self.last_epoch > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma
        self.last_epoch += 1


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ## Train the model using Train set

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
# Initialize the T5 model
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.resize_token_embeddings(len(task_tokenizer))
model.to(device) 


# Parameters for training
num_epochs = 500
patience = 15
best_loss = float('inf')
batch_size = 16
best_model_path = "output/cs_t5_large_bestmodel"
patience_counter = 0

# Create DataLoader for training data
train_dataloader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define checkpoint file path
checkpoint_file = 'output/checkpoint_t5_large'

# Load checkpoint if it exists
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = AdamW(model.parameters(), lr=1e-4)  # Re-initialize optimizer if needed
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    patience_counter = checkpoint['patience_counter']  # Loaded from checkpoint
    print(f"Resuming from epoch {start_epoch}. Epoch {start_epoch - 1}'s best loss: {best_loss} and patience counter: {patience_counter}")
else:
    start_epoch = 0  # Start fresh if no checkpoint
    optimizer = AdamW(model.parameters(), lr=1e-4)
    patience_counter = 0  # Initialize patience counter

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Reduce LR by 0.1 every 30 epochs

# Early stopping
early_stopping = EarlyStopping(patience=patience)

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch}, Average Loss: {avg_loss}")

    # Learning rate scheduling
    scheduler.step()  

    # Check if the current avg_loss is better than the best_loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0  # Reset the patience counter when the loss improves
        model.save_pretrained(best_model_path)  # Save the best model
        print(f"Best model saved at {best_model_path}")
    else:
        patience_counter += 1  # Increment patience counter when the loss does not improve

    # Early stopping based on patience
    if patience_counter >= patience:
        print(f"Stopping early after {epoch+1} epochs due to no improvement.")
        break

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'patience_counter': patience_counter,  # Save the updated patience counter
    }, checkpoint_file)





