from CodeSyntaxConcept.core.aligners.custom_aligner import CustomAligner
from CodeSyntaxConcept.core.parsers.tree_sitter_unparser import TreeSitterParser


class ConceptMapper:
    @staticmethod
    def map_ast_families(source_code: str, model_tokenizer, programming_language):
        # model tokenizer output
        model_tokens = model_tokenizer.tokenize(source_code)
        # AST parser
        ast_tokens = TreeSitterParser.tokenize(source_code, programming_language)
        # custom aligner
        custom_aligner = CustomAligner(model_tokens.copy(), ast_tokens.copy())
        custom_aligner.align_tokens()
        custom_aligner_output = custom_aligner.get_model_tokens_dataframe()
        return custom_aligner_output
