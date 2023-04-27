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

parent_node_types_path = "/workspaces/CodeSyntaxConcept/data/scripts/parent_node_types.csv"
child_node_types_path = "/workspaces/CodeSyntaxConcept/data/scripts/child_node_types.csv"

### PARAMETERS
#checkpoint = "EleutherAI/gpt-neo-125m" #c1
#checkpoint = "EleutherAI/gpt-neo-2.7B" #c2
#checkpoint = "Salesforce/codegen-2B-nl" #c3
#checkpoint = "Salesforce/codegen-350M-nl" #c5
#checkpoint = "Salesforce/codegen-2B-nl" #c6
#checkpoint = "codeparrot/codeparrot-small-multi" #c9
#checkpoint = "Salesforce/codegen-350M-multi" #c10
#checkpoint = "Salesforce/codegen-2B-multi" #c11
#checkpoint = "codeparrot/codeparrot-small" #c14
#checkpoint = "codeparrot/codeparrot" #c15
#checkpoint = "Salesforce/codegen-350M-mono" #c16
checkpoint = "Salesforce/codegen-2B-mono" #c17

#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c1.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c2.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c3.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c5.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c6.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c9.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c10.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c11.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c14.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c15.csv"
#aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c16.csv"
aggregates_path = "/workspaces/CodeSyntaxConcept/scripts_output/out_astevalverticalfiltered_c17.csv"

#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c1.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c2.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c3.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c5.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c6.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c9.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c10.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c11.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c14.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c15.csv"
#output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c16.csv"
output_path = "/workspaces/CodeSyntaxConcept/data/ds_processed_logits_local/out_astevalverticalfiltered_c17.csv"


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
#most_frequent_leaves = ['identifier', '.', '(', ')', ',', '=', 'string',':','[',']','integer']
#most_frequent_parents = ['attribute','expression_statement','argument_list','call','assignment','comparison_operator', 'if_statement','return_statement','for_statement', 'parameters', 'function_definition']
#concepts = most_frequent_leaves + most_frequent_parents
concepts = ['for_statement', 'while_statement', 'return_statement', ']', ')', 'if_statement', 'comparison_operator', 'boolean_operator', 'for_in_clause', 'if_clause', 'list_comprehension', 'lambda', 'identifier' ,'string']
df_concept_embeddings = pd.DataFrame([], columns= concepts)
for binded_tree in df_actual_ntp['binded_tree']:
    df_concept_embeddings.loc[len(df_concept_embeddings.index)] = get_concept_embeddings(binded_tree, concepts)

df_concept_embeddings = df_concept_embeddings.set_index(df_actual_ntp.index)
for concept in concepts:
    df_actual_ntp[str(concept)] = df_concept_embeddings[str(concept)]

print(df_actual_ntp.head())
### SAVE THE OUTPUT
df_actual_ntp.to_csv(output_path)

