from multiprocess import set_start_method
import pandas as pd
from CodeSyntaxConcept.core.parsers.tree_sitter_parser import TreeSitterParser
from transformers import PreTrainedTokenizerFast
import numpy as np

class CodeSearchNet:

    SMALL = 200
    MEDIUM = 500
    LARGE = 1000

    @staticmethod
    def get_test_sets(test_set, with_ranks=False, num_proc=1):
        """deprecated"""
        if with_ranks:
            set_start_method("spawn")
        # perform tasks
        testset_small = test_set.filter(lambda sample:
                                        len(sample['func_code_tokens']) <= CodeSearchNet.SMALL,
                                        num_proc=num_proc)
        testset_medium = test_set.filter(lambda sample:
                                         CodeSearchNet.SMALL < len(
                                             sample['func_code_tokens']) <= CodeSearchNet.MEDIUM,
                                         num_proc=num_proc)
        testset_large = test_set.filter(lambda sample:
                                        CodeSearchNet.MEDIUM < len(
                                            sample['func_code_tokens']) <= CodeSearchNet.LARGE,
                                        num_proc=num_proc)
        return testset_small, testset_medium, testset_large

    @staticmethod
    def get_test_sets(test_set, language, max_token_number, model_tokenizer: PreTrainedTokenizerFast, with_ranks=False, num_proc=1):
        subset = test_set.filter(lambda sample: True if sample['language']== language 
            and len(sample['func_code_tokens']) <= max_token_number
            and len(model_tokenizer(sample['whole_func_string'])) <= max_token_number
            else False, num_proc=num_proc)
        return subset
        #return test_set.filter(lambda sample: True if sample['language']== language 
        #    and len(sample['func_code_tokens']) <= max_token_number else False, num_proc=num_proc)


    @staticmethod
    def get_sub_set_test_set(test_set, test_size:int):
        sub_samples = []
        for sample in test_set:
            sub_samples.append(sample)
            if len(sub_samples)>=test_size:
                break
        return sub_samples
        

    @staticmethod
    def count_source_code_ast_type_frequency(source_code: str, language: str):
        code_sample_node_types = TreeSitterParser.process_source_code(source_code, language)
        token_counts = code_sample_node_types[code_sample_node_types.columns[0]].value_counts(dropna=True, sort=True)
        node_type_counts = code_sample_node_types[code_sample_node_types.columns[1]].value_counts(dropna=True, sort=True)
        parent_node_type_counts = code_sample_node_types[code_sample_node_types.columns[2]].value_counts(dropna=True, sort=True)
        return token_counts, node_type_counts, parent_node_type_counts

    @staticmethod
    def create_ast_concepts_dataframe_from_testset(test_set, model_tokenizer: PreTrainedTokenizerFast):
        test_set_concepts = pd.DataFrame([], columns=['whole_func_string', 'ast_concepts', 'model_tokenizer_concepts'])
        for test_sample in test_set: 
            ast_concepts = TreeSitterParser.process_source_code(test_sample['whole_func_string'], test_sample['language']).to_numpy()
            tokenizer_concepts =  TreeSitterParser.process_model_source_code(test_sample['whole_func_string'], test_sample['language'], model_tokenizer).to_numpy()
            test_set_concepts.loc[len(test_set_concepts.index)] = (test_sample['whole_func_string'], ast_concepts, tokenizer_concepts)
        return test_set_concepts
        
    @staticmethod
    def transform_code_counts_to_dataframe(total_token_counts: pd.Series, total_node_type_counts: pd.Series, total_parent_node_type_counts: pd.Series):
        ## total token frequency counts
        df_total_token_counts = total_token_counts.to_frame()
        df_total_token_counts.reset_index(inplace=True)
        df_total_token_counts.columns = ['token', 'counts']
        ## total node type frequency counts
        df_total_node_type_counts = total_node_type_counts.to_frame()
        df_total_node_type_counts.reset_index(inplace=True)
        df_total_node_type_counts.columns = ['node_type', 'counts']
        ## total parent node type frequency counts
        df_total_parent_node_type_counts = total_parent_node_type_counts.to_frame()
        df_total_parent_node_type_counts.reset_index(inplace=True)
        df_total_parent_node_type_counts.columns = ['parent_type', 'counts']
        return df_total_token_counts, df_total_node_type_counts, df_total_parent_node_type_counts
    
    @staticmethod
    def add_count_average(count_dataframe: pd.DataFrame):
        total_count = count_dataframe['counts'].sum()
        average_column = [(row['counts']/total_count)*100 for index, row in count_dataframe.iterrows()]
        count_dataframe['avg'] = average_column
        return count_dataframe

    @staticmethod
    def count_ast_type_frequency(test_set):
        total_token_counts = pd.Series(dtype='float64')
        total_node_type_counts = pd.Series(dtype='float64')
        total_parent_node_type_counts = pd.Series(dtype='float64')
        for code_sample in test_set:
            token_counts, node_type_counts, parent_node_type_counts = CodeSearchNet.count_source_code_ast_type_frequency(code_sample['whole_func_string'], code_sample['language'])
            total_token_counts = total_token_counts.add(token_counts, fill_value=0)
            total_node_type_counts = total_node_type_counts.add(node_type_counts, fill_value=0)
            total_parent_node_type_counts = total_parent_node_type_counts.add(parent_node_type_counts, fill_value=0) 
        return CodeSearchNet.transform_code_counts_to_dataframe(total_token_counts, total_node_type_counts, total_parent_node_type_counts)