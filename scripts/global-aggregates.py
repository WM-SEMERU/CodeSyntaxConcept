## Imports 
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


## Parameters 
language = "python"
checkpoint = "EleutherAI/gpt-neo-125M"
parent_node_types_path = "output/nodes/parent_node_types.csv"
child_node_types_path = "output/nodes/child_node_types.csv"
aggregates_path = "output/aggregation_function/codesearch_tesbed_EleutherAI-gpt-neo-125M_10000_aggregated.csv"
output_path = "output/global_aggregation/codesearch_tesbed_EleutherAI-gpt-neo-125M_10000_global.csv"

tokenizer = CodeTokenizer.from_pretrained(checkpoint, language)

## Parent Nodes
parent_node_types = pd.read_csv(parent_node_types_path, index_col=0)
parent_node_types = set(parent_node_types['parent_type'])

## Children Nodes
child_node_types = pd.read_csv(child_node_types_path, index_col=0)
child_node_types = set(child_node_types['child_type'])

## Load Aggregates
df_actual_ntp = pd.read_csv(aggregates_path, index_col=0)

##Convert JSON do Dict
df_actual_ntp['binded_tree'] = df_actual_ntp['binded_tree'].map(lambda binded_tree: json.loads(binded_tree))

df_actual_ntp.head()

## Local Analysis 
def traverse_tree_and_collect_stds(node: dict, node_types_list: list, std_field: str):
    if node[std_field] is not None:
        node_types_list[tokenizer.node_types.index(node['type'])] = node_types_list[tokenizer.node_types.index(node['type'])] + [node[std_field]]
    for child in node['children']:
        traverse_tree_and_collect_stds(child, node_types_list, std_field)

def add_statistic_column(std_field, dataframe):
    concept_probs = []
    for tree in dataframe['binded_tree']:
        node_types_list = [[] for type in tokenizer.node_types]
        traverse_tree_and_collect_stds(tree, node_types_list, std_field)
        snippet_type_list = []
        for type_index, node_values in enumerate(node_types_list):
            if len(node_values)>0: 
                snippet_type_list.append((tokenizer.node_types[type_index], node_values))
        concept_probs.append(snippet_type_list)
    dataframe['concept_'+std_field] =  concept_probs

add_statistic_column('median_prob', df_actual_ntp)
add_statistic_column('min_prob', df_actual_ntp)
add_statistic_column('max_prob', df_actual_ntp)
df_actual_ntp.head()

## Gobal Analysis 
def collect_global_std(std_field, dataframe):
    node_types_list = [[] for type in tokenizer.node_types]
    for tree in dataframe['binded_tree']:
        traverse_tree_and_collect_stds(tree, node_types_list, std_field)
    return node_types_list

concept_median_prob_list = collect_global_std('median_prob', df_actual_ntp)
concept_min_prob_list = collect_global_std('min_prob', df_actual_ntp)
concept_max_prob_list = collect_global_std('max_prob', df_actual_ntp)

global_concept_dataframe = pd.DataFrame([], columns=['ast_element', 'node_type' ,'concept_median_prob', 'concept_min_prob','concept_max_prob'])
for concept_idx in range(0,len(tokenizer.node_types)):
    if(len(concept_median_prob_list[concept_idx])>0):
        global_concept_dataframe.loc[len(global_concept_dataframe.index)] = [tokenizer.node_types[concept_idx],
                                                                             'parent' if tokenizer.node_types[concept_idx] in parent_node_types else 'leaf',
                                                                             concept_median_prob_list[concept_idx], 
                                                                             concept_min_prob_list[concept_idx], 
                                                                             concept_max_prob_list[concept_idx]]

print(global_concept_dataframe.head())
global_concept_dataframe.to_csv(output_path)