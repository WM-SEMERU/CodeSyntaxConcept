from datasets import load_dataset 
import CodeSyntaxConcept.utils as utils
from CodeSyntaxConcept.tokenizer import CodeTokenizer
from CodeSyntaxConcept.parser import TreeSitterParser
import pandas as pd


#### PARAMETERS #####
#checkpoint = "EleutherAI/gpt-neo-125M"
#checkpoint = "EleutherAI/gpt-neo-1.3B"
#checkpoint = "EleutherAI/gpt-neo-2.7B"
#checkpoint = "microsoft/CodeGPT-small-py"
#checkpoint = "microsoft/CodeGPT-small-py-adaptedGPT2"
#checkpoint = "Salesforce/codegen-16B-multi"
#checkpoint = "Salesforce/codegen-6B-multi"
#checkpoint = "Salesforce/codegen-2B-multi"
checkpoint = "himanshu-dutta/pycoder-gpt2"

number_of_samples = 10000
language = "python"
save_path = "../experimental_notebooks/exploratory_experiments"

############## LOAD SAMPLES ##############
test_set = load_dataset("code_search_net", split='test')
test_set = test_set.filter(lambda sample: True if 
                sample['language']== language 
            else False, num_proc=1)
test_set = utils.get_random_sub_set_test_set(test_set, number_of_samples)

############## MAPPING CONCEPTS ###########
parser = TreeSitterParser(CodeTokenizer.from_pretrained(checkpoint,language))
test_set_node_counts = [()]*len(parser.tokenizer.node_types)
for sample in test_set:
    tree = parser.tokenizer.parser.parse(bytes(sample['whole_func_string'], "utf8"))
    for ast_element in parser.tokenizer.node_types:
        ast_element_ocurrences = []
        utils.find_nodes(tree.root_node, ast_element, ast_element_ocurrences)
        test_set_node_counts[parser.tokenizer.node_types.index(ast_element)] = test_set_node_counts[parser.tokenizer.node_types.index(ast_element)] + tuple([len(ast_element_ocurrences)])


############## COUNTS DATAFRAME ###########
node_counts_dataframe = pd.DataFrame([], columns=['ast_element', 'counts'])
for node_id, node_counts in enumerate(test_set_node_counts):
    node_counts_dataframe.loc[len(node_counts_dataframe.index)] = [parser.tokenizer.node_types[node_id], node_counts]

node_counts_dataframe.head()

############## SAVE RESULTS ##############
node_counts_dataframe.to_csv(save_path+"/ea_"+checkpoint.replace("/","-")+".csv")