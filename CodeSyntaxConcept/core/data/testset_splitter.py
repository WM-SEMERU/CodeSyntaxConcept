from multiprocess import set_start_method

class CodeSearchNetSplitter:
    SMALL = 200
    MEDIUM = 500
    LARGE = 1000

    @staticmethod
    def get_test_sets(test_set, with_ranks: False, num_proc=1):
        if with_ranks:
            set_start_method("spawn")
        # perform tasks
        testset_small = test_set.filter(lambda sample:
                                        len(sample['func_code_tokens']) <= CodeSearchNetSplitter.SMALL,
                                        num_proc=num_proc)
        testset_medium = test_set.filter(lambda sample:
                                         CodeSearchNetSplitter.SMALL < len(
                                             sample['func_code_tokens']) <= CodeSearchNetSplitter.MEDIUM,
                                         num_proc=num_proc)
        testset_large = test_set.filter(lambda sample:
                                        CodeSearchNetSplitter.MEDIUM < len(
                                            sample['func_code_tokens']) <= CodeSearchNetSplitter.LARGE,
                                        num_proc=num_proc)
        return testset_small, testset_medium, testset_large
