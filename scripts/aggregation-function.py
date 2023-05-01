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

language = "python"
### PARAMETERS
#checkpoint = "EleutherAI/gpt-neo-125m" #c1
#checkpoint = "EleutherAI/gpt-neo-1.3B" #c2
checkpoint = "EleutherAI/gpt-neo-2.7B" #c3 
#checkpoint = "Salesforce/codegen-350M-nl" #c5
#checkpoint = "Salesforce/codegen-2B-nl" #c6
#checkpoint = "codeparrot/codeparrot-small-multi" #c9
#checkpoint = "Salesforce/codegen-350M-multi" #c10
#checkpoint = "Salesforce/codegen-2B-multi" #c11
#checkpoint = "codeparrot/codeparrot-small" #c14
#checkpoint = "codeparrot/codeparrot" #c15
#checkpoint = "Salesforce/codegen-350M-mono" #c16
#checkpoint = "Salesforce/codegen-2B-mono" #c17


#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c1.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c2.csv"
file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c3.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c5.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c6.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c9.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c10.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c11.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c14.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c15.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c16.csv"
#file_path = "/workspaces/CodeSyntaxConcept/data/ds_raw_logits/out_astevalverticalfiltered_c17.csv"


#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c1.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c2.csv"
output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c3.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c5.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c6.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c9.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c10.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c11.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c14.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c15.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c16.csv"
#output_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c17.csv"

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
        
encoding = tokenizer.tokenizer(df_actual_ntp.iloc[0]['code'], return_offsets_mapping=True)
assert len(eval(df_actual_ntp.iloc[0]['ids'])) == len(encoding['input_ids'])

binded_tree_col = []
for index, row in df_actual_ntp.iterrows():
    tree = tokenizer.parser.parse(bytes(row['code'], "utf8"))
    encoding = tokenizer.tokenizer(row['code'], return_offsets_mapping=True)
    actual_logits = eval(row['actual_prob'])
    actual_logits.insert(0,(tokenizer.tokenizer.decode(eval(row['ids'])[0]),'FIRST_TOKEN'))
    binded_tree = bind_bpe_tokens(tree.root_node, encoding, actual_logits, row['code'].split('\n'))
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
