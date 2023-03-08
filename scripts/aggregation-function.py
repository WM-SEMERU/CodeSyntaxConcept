##### IMPORTS 
import pandas as pd
import os
import time
import numpy as np
import torch
from datasets import load_dataset 
from CodeSyntaxConcept.tokenizer import CodeTokenizer
import CodeSyntaxConcept.utils as utils
from statistics import mean, median
import json

### PARAMETERS
checkpoint = "EleutherAI/gpt-neo-2.7B"
file_path = "output/raw_logits/out_codesearch_tesbed_EleutherAI-gpt-neo-2.7B_10000.csv"
language = "python"
output_path = "output/aggregation_function/codesearch_tesbed_EleutherAI-gpt-neo-2.7B_10000_aggregated.csv"

### TOKENIZER
tokenizer = CodeTokenizer.from_pretrained(checkpoint, language)

### ACTUAL TOKEN PREDICTIONS

df_actual_ntp = pd.read_csv(file_path, index_col=0)

#### TOKEN BINDINGS

def bind_bpe_tokens(
    node,              #Tree sitter ast tree
    encoding,          #Token encoding
    actual_probs,      #Actual probabilities
    lines              #Source code Snippet
): 
    """Traverses the tree and bind the leaves with the corresponding node"""
    tree_node = {}
    tree_node['type'] = node.type
    tree_node['children'] = []
    tree_node['bindings'] = []

    node_span = [utils.convert_to_offset(node.start_point, lines), utils.convert_to_offset(node.end_point, lines)]
    for encoding_index, token_span in enumerate(encoding.offset_mapping):
        if (node_span[0] <= token_span[0] and token_span[0] < node_span[1]) \
        or (node_span[0] < token_span[1] and token_span[1] <= node_span[1]) \
        or (node_span[0] >= token_span[0] and token_span[1] >= node_span[1]) :
            tree_node['bindings'].append(actual_probs[encoding_index])
    
    for n in node.children:
        tree_node['children'].append(bind_bpe_tokens(n, encoding, actual_probs, lines))

    return tree_node
        
encoding = tokenizer.tokenizer(df_actual_ntp.iloc[0]['whole_func_string'], return_offsets_mapping=True)
assert len(eval(df_actual_ntp.iloc[0]['model_input_ids'])) == len(encoding['input_ids'])

binded_tree_col = []
for index, row in df_actual_ntp.iterrows():
    tree = tokenizer.parser.parse(bytes(row['whole_func_string'], "utf8"))
    encoding = tokenizer.tokenizer(row['whole_func_string'], return_offsets_mapping=True)
    actual_logits = eval(row['actual_prob_case'])
    actual_logits.insert(0,(tokenizer.tokenizer.decode(eval(row['model_input_ids'])[0]),'FIRST_TOKEN'))
    binded_tree = bind_bpe_tokens(tree.root_node, encoding, actual_logits, row['whole_func_string'].split('\n'))
    binded_tree_col.append(binded_tree)
df_actual_ntp['binded_tree'] = binded_tree_col

def process_bindings(
    node: dict,     #Binded AST tree with actual probabilities
) -> None:
    node_actual_probs = [binding[1] for binding in node['bindings'] if isinstance(binding[1], float)]
    node['median_prob'] = node['max_prob'] = node['min_prob'] = node['avg_prob'] =  node['std'] = None
    ## len is zero if node correspond to FIRST_TOKEN = 'def' 
    if(len(node_actual_probs) > 0):
        ## BOOTSTRAPPING-> 
        node_actual_probs = utils.bootstrapping(node_actual_probs, np.mean, size=100).tolist()
        ##
        node['median_prob'] = median(node_actual_probs) 
        node['max_prob'] = max(node_actual_probs) 
        node['min_prob'] = min(node_actual_probs)
        node['avg_prob'] = mean(node_actual_probs)
        node['std'] = np.std(node_actual_probs)
    for child in node['children']:
        process_bindings(child)

df_actual_ntp['binded_tree'].apply(lambda binded_tree: process_bindings(binded_tree))
print(df_actual_ntp.iloc[0]['binded_tree']['children'][0]['children'][1]['median_prob'])
print(df_actual_ntp.iloc[1]['binded_tree']['children'][0]['children'][1]['median_prob'])

print(df_actual_ntp.head())


##Convert to JSON Object
df_actual_ntp['binded_tree'] = df_actual_ntp['binded_tree'].map(lambda binded_tree: json.dumps(binded_tree))


### SAVE THE OUTPUT
df_actual_ntp.to_csv(output_path)
