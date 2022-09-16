from core.aligners.custom_aligner import CustomAligner
from core.parsers.tree_sitter_unparser import TreeSitterParser


class ConceptMapper:
    @staticmethod
    def map_ast_families(source_code: str, model_tokenizer, programming_language):
        # model tokenizer output
        model_tokens = model_tokenizer.tokenize(source_code)
        # AST parser
        tree_sitter_tokenizer = TreeSitterParser(source_code, programming_language)
        tree_sitter_tokenizer.tokenize()
        ast_tokens = tree_sitter_tokenizer.get_tokens()
        # custom aligner
        custom_aligner = CustomAligner(model_tokens.copy(), ast_tokens.copy())
        custom_aligner.align_tokens()
        custom_aligner_output = custom_aligner.get_model_tokens_dataframe()
        return custom_aligner_output
