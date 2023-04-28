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


aggregates_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/aggregation_function/out_astevalverticalfiltered_c2.csv"
#aggregates_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/aggregation_function/out_astevalverticalfiltered_c3.csv"
#aggregates_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/aggregation_function/out_astevalverticalfiltered_c6.csv"

embedding_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/embedding/out_astevalverticalfiltered_c2_bk.csv"
#embedding_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/embedding/out_astevalverticalfiltered_c3_bk.csv"
#embedding_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/embedding/out_astevalverticalfiltered_c6_bk.csv"

output_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/embedding/out_astevalverticalfiltered_c2.csv"
#output_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/embedding/out_astevalverticalfiltered_c3.csv"
#output_path = "/scratch1/svelascodimate/CodeSyntaxConcept/scripts/output/embedding/out_astevalverticalfiltered_c6.csv"

## Load Aggregates
df_actual_ntp = pd.read_csv(aggregates_path, index_col=0)
df_concept_embeddings = pd.read_csv(embedding_path, index_col=0)

## Load Embeddings

most_frequent_leaves = ['identifier', '.', '(', ')', ',', '=', 'string',':','[',']','integer']
most_frequent_parents = ['attribute','expression_statement','argument_list','call','assignment','comparison_operator', 'if_statement','return_statement','for_statement', 'parameters', 'function_definition']
concepts = most_frequent_leaves + most_frequent_parents

df_concept_embeddings = df_concept_embeddings.set_index(df_actual_ntp.index)
for concept in concepts:
    df_actual_ntp[str(concept)] = df_concept_embeddings[str(concept)]

df_actual_ntp.to_csv(output_path)