# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/evaluator.ipynb.

# %% auto 0
__all__ = ['Evaluator']

# %% ../nbs/evaluator.ipynb 2
import CodeSyntaxConcept

from .tokenizer import CodeTokenizer
from .parser import TreeSitterParser
import CodeSyntaxConcept.utils as utils
import pandas as pd

# %% ../nbs/evaluator.ipynb 4
class Evaluator:

    def __init__(self, checkpoint: str, language):
        self.tokenizer = CodeTokenizer.from_pretrained(checkpoint, language)
        self.parser = TreeSitterParser(self.tokenizer)
    
    def __call__(self, test_set):
        test_set_concepts = pd.DataFrame([], columns=['whole_func_string', 'ast_concepts', 'model_tokenizer_concepts', 'model_input_ids', 'model_total_input_ids'])
        for test_sample in test_set: 
            ast_concepts = self.parser.process_source_code(test_sample['whole_func_string'])
            source_code_encoding, tokenizer_concepts =  self.parser.process_model_source_code(test_sample['whole_func_string'])
            test_set_concepts.loc[len(test_set_concepts.index)] = (test_sample['whole_func_string'], ast_concepts, tokenizer_concepts, source_code_encoding['input_ids'], len(source_code_encoding['input_ids']))
        return test_set_concepts