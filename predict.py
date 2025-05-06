
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
import os
warnings.filterwarnings("ignore")

model_name = "t5-large"
best_model_path = "output/cs_t5_large_bestmodel"
save_prediction_path = 'output/cs_t5_large_predictions.json'
dataset_dict = DatasetDict()
# Load the dataset from disk
dataset_dict['train'] = load_from_disk('data/cs_paper_author_over2pub/dataset_directory/train')
dataset_dict['test'] = load_from_disk('data/cs_paper_author_over2pub/dataset_directory/test')


train_data = dataset_dict['train']
test_data = dataset_dict['test']

def tokenize_data_with_id(examples, task_tokenizer, team_tokenizer, max_input_length=512, max_target_length=10):
    # Tokenize task description (input)
    inputs = task_tokenizer(
        examples['input'], 
        max_length=max_input_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )
    
    # Tokenize team author IDs (target)
    targets = []
    for team_str in examples['team']:
        author_list = team_tokenizer.string_to_list(team_str)  # Convert team string to list
        tokenized_team = team_tokenizer.convert_tokens_to_ids(author_list)  # Convert team list to token IDs
        
        # Pad or truncate the target author IDs to ensure uniform length
        if len(tokenized_team) < max_target_length:
            tokenized_team += [team_tokenizer.token_to_id['<pad>']] * (max_target_length - len(tokenized_team))  # Pad to max length
        targets.append(tokenized_team[:max_target_length])  # Truncate if too long
    
    return {
        'id': examples['id'],  # Include the id field
        'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
        'attention_mask': inputs['attention_mask'].squeeze(0),  # Remove batch dimension
        'labels': torch.tensor(targets, dtype=torch.long)  # Ensure labels are converted to a tensor
    }

def collate_fn_validation(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    ids = [item['id'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'id': ids
    }


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

## Task tokenizer
task_tokenizer = T5Tokenizer.from_pretrained(model_name)

## Team tokenizer
cs_paper_author = pd.read_csv("./output/unique_author_published_2plus_cs_paper.csv")
#display(cs_paper_author.head())
list_cs_paper_author = cs_paper_author["author_id"].unique()
print(len(list_cs_paper_author))

# Initialize the team tokenizer
team_tokenizer = SimpleTeamTokenizer(list_cs_paper_author)

# Tokenize test data
test_tokenized = test_data.map(lambda x: tokenize_data_with_id(x, task_tokenizer, team_tokenizer), batched=True)

# Create a DataLoader for validation
test_dataloader = DataLoader(test_tokenized, batch_size=8, collate_fn=collate_fn_validation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load model
model = T5ForConditionalGeneration.from_pretrained(best_model_path)
model.to(device)

# print model size
total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters:", total_params)

# evaluate
model.eval()
predictions = []
true_labels = []
ids = []

# Disable gradient calculation for evaluation
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Generate predictions from the model
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=10, 
            min_length=2, 
            num_beams=5, 
            do_sample=False
        )
        
        # Convert predicted token IDs to author IDs, skipping <pad> tokens
        predicted_author_ids = [
            team_tokenizer.convert_ids_to_tokens(pred.tolist(), skip_special_tokens=True) for pred in outputs
        ]
        
        # Remove duplicate author IDs while keeping the order
        unique_predicted_author_ids = [list(dict.fromkeys(pred)) for pred in predicted_author_ids]



        # Convert true labels to author IDs, skipping <pad> tokens
        true_author_ids = [
            team_tokenizer.convert_ids_to_tokens(true_label.tolist(), skip_special_tokens=True) for true_label in batch['labels']
        ]
        
        # Collect predictions, true labels, and ids
        predictions.extend(unique_predicted_author_ids)
        true_labels.extend(true_author_ids)
        ids.extend(batch['id'])

# At this point, `predictions`, `true_labels`, and `ids` are lists of corresponding values.
# Now calculate recall@10
recall_mean, recalls_per_sample = r_at_k(predictions, true_labels, k=10)
precision_mean, precision_per_sample = p_at_k(predictions, true_labels, k=10)


# Combine results: Map IDs to predictions and true labels
result = [{"id": id_, "y_true": true, "y_pred": pred, "recall@10": recall, "precision@10": precision

           } 
          for id_, true, pred, recall, precision in zip(
              ids, true_labels, predictions, recalls_per_sample, precision_per_sample)]

# # Display results for inspection
# for item in result:
#     print(f"ID: {item['id']}, True Team: {item['y_true']}, Predicted Team: {item['y_pred']}, Recall@10: {item['recall@10']}, Precision@10: {item['precision@10']}")



# Create a list of dictionaries for each prediction
data = [{'id': id_, 'true_labels': true, 'predictions': pred} for id_, true, pred in zip(ids, true_labels, predictions)]

# Save to JSON file
with open(save_prediction_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Predictions saved")

