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

parent_node_types_path = "/workspaces/CodeSyntaxConcept/data/scripts/parent_node_types.csv"
child_node_types_path = "/workspaces/CodeSyntaxConcept/data/scripts/child_node_types.csv"

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


#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c1.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c2.csv"
aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c3.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c5.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c6.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c9.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c10.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c11.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c14.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c15.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c16.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c17.csv"


#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c1.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c2.csv"
output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c3.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c5.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c6.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c9.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c10.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c11.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c14.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c15.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c16.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_global/out_astevalverticalfiltered_c17.csv"


tokenizer = CodeTokenizer.from_pretrained(checkpoint, language)

## Parent Nodes
parent_node_types = pd.read_csv(parent_node_types_path, index_col=0)
parent_node_types = set(parent_node_types['parent_type'])

print('############# READING FILES #############')

## Children Nodes
child_node_types = pd.read_csv(child_node_types_path, index_col=0)
child_node_types = set(child_node_types['child_type'])

## Load Aggregates
df_actual_ntp = pd.read_csv(aggregates_path, index_col=0)

print(df_actual_ntp.info())

print('############# PARSING AST #############')

##Convert JSON do Dict
df_actual_ntp['binded_tree'] = df_actual_ntp['binded_tree'].map(lambda binded_tree: json.loads(binded_tree))

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

print('############# ADDING STATISTICS #############')

add_statistic_column('median_prob', df_actual_ntp)
add_statistic_column('min_prob', df_actual_ntp)
add_statistic_column('max_prob', df_actual_ntp)

print('############# COLLECTING STATISTICS #############')
## Gobal Analysis
global_concept_dataframe = pd.DataFrame([], columns=['ast_element', 'node_type' ,'concept_median_prob', 'concept_min_prob','concept_max_prob'])
for concept_idx in range(0,len(tokenizer.node_types)):
    global_concept_dataframe.loc[len(global_concept_dataframe.index)] = [tokenizer.node_types[concept_idx],
                                                                             'parent' if tokenizer.node_types[concept_idx] in parent_node_types else 'leaf',
                                                                             [], 
                                                                             [], 
                                                                             []]
for index, tree in enumerate(df_actual_ntp['binded_tree']):
    if index%1000 == 0:
        print(index)
    concept_median_prob_list = [[] for type in tokenizer.node_types]
    concept_min_prob_list = [[] for type in tokenizer.node_types]
    concept_max_prob_list = [[] for type in tokenizer.node_types]
    traverse_tree_and_collect_stds(tree, concept_median_prob_list, 'median_prob')
    traverse_tree_and_collect_stds(tree, concept_min_prob_list, 'min_prob')
    traverse_tree_and_collect_stds(tree, concept_max_prob_list, 'max_prob')
    for concept_idx in range(0,len(tokenizer.node_types)):
        global_concept_dataframe.at[concept_idx, 'concept_median_prob'] = global_concept_dataframe['concept_median_prob'][concept_idx] + concept_median_prob_list[concept_idx]
        global_concept_dataframe.at[concept_idx, 'concept_min_prob'] = global_concept_dataframe['concept_min_prob'][concept_idx] + concept_min_prob_list[concept_idx]
        global_concept_dataframe.at[concept_idx, 'concept_max_prob'] = global_concept_dataframe['concept_max_prob'][concept_idx] + concept_max_prob_list[concept_idx]

global_concept_dataframe = global_concept_dataframe.drop(global_concept_dataframe[global_concept_dataframe['concept_median_prob'].map(len) == 0].index)
global_concept_dataframe['model'] = checkpoint

print('############# SAVING FILE #############')

print(global_concept_dataframe.head())
global_concept_dataframe.to_csv(output_path)