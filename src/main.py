from transformers import AutoTokenizer, AutoModelWithLMHead

from src.aligners.custom_aligner import CustomAligner
from src.aligners.needleman_wunch import NeedlemanWunch
from src.parsers.ast_unparser import UnparserTokenizer

import ast

if __name__ == '__main__':
    # Model
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    # Define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # source code
    source_code = 'def testing(a,b):\n variable_1=5656556\n variable_2=5656556\n while true: return a*b'

    # unparser tokenizer
    unparser_tokenizer = UnparserTokenizer()
    parsed_ast = ast.parse(source_code, mode='exec')

    # deep model output
    model_tokens = tokenizer.tokenize(source_code)
    # model_tokens = ['def','testing','a','b']

    # python unparser tokenizer output
    ast_tokens = unparser_tokenizer.find_tokens(parsed_ast)

    # neddleman wunch to find association
    needleman_wunch = NeedlemanWunch(model_tokens.copy(), ast_tokens.copy())
    needleman_wunch.find_optimal_sequence()

    # custon aligner
    custom_aligner = CustomAligner(model_tokens.copy(), ast_tokens.copy())
    custom_aligner.align_tokens()

    # get results
    needleman_output = needleman_wunch.get_model_tokens_queue()
    custom_aligner_output = custom_aligner.get_model_tokens_queue()

    print('\n-------------------- NEEDLEMAN WUNCH ALIGNER -------------------\n')
    for token in needleman_output:
        print(token)

    print('\n------------------------- CUSTOM ALIGNER -----------------------\n')
    for token in custom_aligner_output:
        print(token)