from datasets import load_dataset    
import pandas as pd

import CodeSyntaxConcept.utils as utils
from CodeSyntaxConcept.evaluator import Evaluator


######### YOU NEED TO SET THIS FIRST #########
#checkpoint = "EleutherAI/gpt-neo-125M"
#checkpoint = "EleutherAI/gpt-neo-1.3B"
#checkpoint = "EleutherAI/gpt-neo-2.7B"
#checkpoint = "microsoft/CodeGPT-small-py"
#checkpoint = "microsoft/CodeGPT-small-py-adaptedGPT2"
#checkpoint = "Salesforce/codegen-16B-multi"
#checkpoint = "Salesforce/codegen-6B-multi"
#checkpoint = "Salesforce/codegen-2B-multi"
checkpoint = "himanshu-dutta/pycoder-gpt2"

language = "python"
maximun_number_of_samples = 10000
save_path = "output/"

######### YOU NEED TO SET THIS FIRST #########
evaluator = Evaluator(checkpoint, language)
print('maximun number of tokens per sequence: '+str(evaluator.tokenizer.tokenizer.max_len_single_sentence))

test_set = utils.get_random_sub_set_test_set(utils.get_test_sets(load_dataset("code_search_net", split='test'), language, evaluator.tokenizer.tokenizer.max_len_single_sentence, evaluator.tokenizer), maximun_number_of_samples)
testset_concepts = evaluator(test_set)

print(testset_concepts.describe())
testset_concepts.to_csv(save_path+"codesearch_tesbed_"+checkpoint.replace("/","-")+"_"+str(maximun_number_of_samples)+".csv")

