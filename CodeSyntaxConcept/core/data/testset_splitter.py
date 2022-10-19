import torch
from datasets import load_dataset
from multiprocess import set_start_method


class CodeSearchNetSplitter:

    SMALL = 200
    MEDIUM = 500
    LARGE = 1000

    @staticmethod
    def split():
        test_set = load_dataset("code_search_net", split='test')
        n_gpus = torch.cuda.device_count()
        return CodeSearchNetSplitter.split_testsets(test_set,
                                                    CodeSearchNetSplitter.SMALL,
                                                    CodeSearchNetSplitter.MEDIUM,
                                                    CodeSearchNetSplitter.LARGE,
                                                    with_ranks=(n_gpus > 0),
                                                    num_proc=n_gpus)

    @staticmethod
    def split_testsets(testset, small_size, medium_size, large_size, with_ranks: False, num_proc=1):
        if with_ranks:
            set_start_method("spawn")
        # perform tasks
        testset_small = testset.filter(lambda sample:
                                       len(sample['func_code_tokens']) <= small_size,
                                       num_proc=num_proc)
        testset_medium = testset.filter(lambda sample:
                                        small_size < len(sample['func_code_tokens']) <= medium_size,
                                        num_proc=num_proc)
        testset_large = testset.filter(lambda sample:
                                       medium_size < len(sample['func_code_tokens']) <= large_size,
                                       num_proc=num_proc)
        return testset_small, testset_medium, testset_large
