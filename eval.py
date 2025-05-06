import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pytrec_eval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred', type=str, required=True, help='Path to saved prediction JSON')
args = parser.parse_args()

save_prediction_path = args.pred


# Load predictions
with open(save_prediction_path, 'r') as f:
    data = json.load(f)

print(f"\n========{save_prediction_path}========")
# Extract fields
ids = [item['id'] for item in data]
true_labels = [item['true_labels'] for item in data]
predictions = [item['predictions'] for item in data]


# Prepare pytrec_eval input
qrel = {}
run = {}

for i, (y_true, y_pred) in enumerate(zip(true_labels, predictions)):
    qid = f'q{i}'
    qrel[qid] = {f'd{label}': 1 for label in y_true}
    
    re you.
    run[qid] = {f'd{doc_id}': 1.0 / (rank + 1) for rank, doc_id in enumerate(y_pred)}

# Set of metrics to compute
metrics = [
    'P_10',
    'recall_10', 
    'recip_rank',
    'ndcg_cut_10',
    'map'
]

print("Evaluating with pytrec_eval...")
evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
results = evaluator.evaluate(run)

# Convert results to DataFrame
import pandas as pd
df = pd.DataFrame.from_dict(results, orient='index')
df_mean = df.mean().to_frame(name='mean')

# Show key metrics
print(df_mean.loc[['P_10', 'recall_10', 'recip_rank', 'ndcg_cut_10', 'map']].round(4))

# F1 Score (still using sklearn)
mlb = MultiLabelBinarizer()
true_binarized = mlb.fit_transform(true_labels)
pred_binarized = mlb.transform(predictions)

f1_micro = f1_score(true_binarized, pred_binarized, average='micro')
print(f"Micro F1 Score: {f1_micro:.4f}")

