## Import 
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


#Paramenters 
language = "python"
checkpoint = "EleutherAI/gpt-neo-125M"
parent_node_types_path = "output/nodes/parent_node_types.csv"
child_node_types_path = "output/nodes/child_node_types.csv"
aggregates_path = "output/aggregation_function/codesearch_tesbed_EleutherAI-gpt-neo-125M_10000_aggregated.csv"
output_path = "output/embedding/codesearch_tesbed_EleutherAI-gpt-neo-125M_10000_embeddings.csv"


tokenizer = CodeTokenizer.from_pretrained(checkpoint, language)

## Parent Nodes
parent_node_types = pd.read_csv(parent_node_types_path, index_col=0)
parent_node_types = set(parent_node_types['parent_type'])

## Child Nodes
child_node_types = pd.read_csv(child_node_types_path, index_col=0)
child_node_types = set(child_node_types['child_type'])

## Load Aggregates
df_actual_ntp = pd.read_csv(aggregates_path, index_col=0)

##Convert JSON do Dict
df_actual_ntp['binded_tree'] = df_actual_ntp['binded_tree'].map(lambda binded_tree: json.loads(binded_tree))

df_actual_ntp.head()

## Concept Embeddings 
def get_concept_embeddings(node, concepts):
    def get_concept_bindings(node, concepts, bindings):
        for child in node['children']:
            get_concept_bindings(child, concepts, bindings)
        if node['type'] in concepts:
            bindings[concepts.index(node['type'])].append([prob for token, prob in node['bindings'] if prob != 'FIRST_TOKEN'])
    bindings=[[] for _ in range(len(concepts))]
    get_concept_bindings(node, concepts, bindings)
    embedding = []
    for prob_list in bindings:
        flat_prob_list = [prob for sublist in prob_list for prob in sublist]
        if(len(flat_prob_list) > 0):
            ### BOOTSTRAPPING to calculate median in embeddings.
            flat_prob_list = utils.bootstrapping(flat_prob_list, np.mean, size=100).tolist()
            ###
            embedding.append(median(flat_prob_list))
        else:
            embedding.append(0)
    return embedding
    
#### MOST FREQUENT CONCEPTS - EXPLORATORY ANALYSIS
most_frequent_leaves = ['identifier', '.', '(', ')', ',', '=', 'string',':','[',']','integer']
most_frequent_parents = ['attribute','expression_statement','argument_list','call','assignment','comparison_operator', 'if_statement','return_statement','for_statement', 'parameters', 'function_definition']
concepts = most_frequent_leaves + most_frequent_parents
df_concept_embeddings = pd.DataFrame([], columns= concepts)
for binded_tree in df_actual_ntp['binded_tree']:
    df_concept_embeddings.loc[len(df_concept_embeddings.index)] = get_concept_embeddings(binded_tree, concepts)

print(df_concept_embeddings.head())
### SAVE THE OUTPUT
df_concept_embeddings.to_csv(output_path)

