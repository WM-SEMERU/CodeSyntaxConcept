from multiprocess import set_start_method
import pandas as pd
from CodeSyntaxConcept.core.parsers.tree_sitter_parser import TreeSitterParser

class CodeSearchNet:

    SMALL = 200
    MEDIUM = 500
    LARGE = 1000

    @staticmethod
    def get_test_sets(test_set, with_ranks=False, num_proc=1):
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
    def count_ast_type_frequency(test_set):
        node_type_counts = pd.Series()
        parent_node_type_counts = pd.Series()
        token_counts = pd.Series()
        for code_sample in test_set: 
            code_sample_node_types = TreeSitterParser.process_source_code(code_sample['whole_func_string'], code_sample['language'])
            token_counts = token_counts.add(code_sample_node_types[code_sample_node_types.columns[0]].value_counts(), fill_value=0)
            node_types = code_sample_node_types[code_sample_node_types.columns[1]].value_counts()
            parent_types = code_sample_node_types[code_sample_node_types.columns[2]].value_counts()
            node_type_counts = node_type_counts.add(node_types, fill_value=0)
            parent_node_type_counts = parent_node_type_counts.add(parent_types, fill_value=0)
        return token_counts.rename_axis('token').to_frame('counts'), node_type_counts.rename_axis('node_type').to_frame('counts'), parent_node_type_counts.rename_axis('parent_node_type').to_frame('counts')
