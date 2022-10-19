import json
import os

import pandas as pd
from tree_sitter import Language, Parser


class TreeSitterParser:

    TREE_SITTER_LANGUAGE_PATHS = [
        'tree-sitter-python',
        'tree-sitter-java']

    Language.build_library('build/my-languages.so', TREE_SITTER_LANGUAGE_PATHS)

    all_node_types = {
        p: json.load(open(os.path.join(p, "src", "node-types.json"), "r"))
        for p in TREE_SITTER_LANGUAGE_PATHS
    }

    # python_node_types = get_language_types(all_node_types, 'python')
    @staticmethod
    def get_language_types(all_node_types, lang):
        node_types = [node_type['type'] for node_type in all_node_types['tree-sitter-' + lang]]
        node_subtypes = [
            node_subtype['type']
            for node_type in all_node_types['tree-sitter-' + lang]
            if 'subtypes' in node_type
            for node_subtype in node_type['subtypes']
        ]
        return list(set(node_types + node_subtypes))

    @staticmethod
    def traverse(node, results) -> None:
        if node.type == 'string':
            results.append(node)
            return
        for n in node.children:
            TreeSitterParser.traverse(n, results)
        if not node.children:
            results.append(node)

    @staticmethod
    def get_token_type_with_span(tok_span, nodes, lines):
        def get_node_span(node, blob):
            def convert_to_offset(point, lines):
                row, column = point
                chars_in_rows = sum(map(len, lines[:row])) + row
                chars_in_columns = len(lines[row][:column])
                offset = chars_in_rows + chars_in_columns
                return offset

            start_span = convert_to_offset(node.start_point, lines)
            end_span = convert_to_offset(node.end_point, lines)
            return start_span, end_span

        node_spans = [get_node_span(node, lines) for node in nodes]
        for i, span in enumerate(node_spans):
            if (span[0] <= tok_span[0] and tok_span[0] < span[1]) or (span[0] < tok_span[1] and tok_span[1] <= span[1]):
                return nodes[i].parent.type, nodes[i].type

    @staticmethod
    def process_model_source_code(source_code: str, language: str, model_tokenizer):
        ## Define Tree Sitter Parser
        parser = Parser()
        parser.set_language(Language('build/my-languages.so', language))
        ## Get the AST representation
        ast_representation = parser.parse(bytes(source_code, "utf8"))
        ## Traverse the tree to get an array representation of all the nodes.
        ast_nodes = []
        TreeSitterParser.traverse(ast_representation.root_node, ast_nodes)
        ## Get Tokens from tokenizer
        source_code_tokens = model_tokenizer.encode_plus(source_code, truncation=True)
        ## Create an array of source code lines
        source_code_lines = source_code.split("\n")

        source_code_ast_types = []
        for token_index in range(len(source_code_tokens.input_ids)):
            ## Get Token AST Types (parent and current node)
            node_type_info = TreeSitterParser.get_token_type_with_span(source_code_tokens.token_to_chars(token_index),
                                                                      ast_nodes, source_code_lines)
            parent_node_type = 'None'
            node_type = 'Node'
            if node_type_info is not None:
                parent_node_type, node_type = node_type_info
            ### Store result
            source_code_ast_types.append(
                [model_tokenizer.decode(source_code_tokens.input_ids[token_index]), node_type, parent_node_type])
        return pd.DataFrame(source_code_ast_types, columns=['token', 'node_type', 'parent_node_type'])

    @staticmethod
    def process_source_code(source_code: str, language: str):
        ## Define Tree Sitter Parser
        parser = Parser()
        parser.set_language(Language('build/my-languages.so', language))
        ## Get the AST representation
        ast_representation = parser.parse(bytes(source_code, "utf8"))
        ## Traverse the tree to get an array representation of all the nodes.
        ast_nodes = []
        TreeSitterParser.traverse(ast_representation.root_node, ast_nodes)
        ## Retrieve node information
        source_code_ast_types = []
        for node_index, node in enumerate(ast_nodes):
            source_code_ast_types.append([node.text.decode("utf-8"), node.type, node.parent.type])
        return pd.DataFrame(source_code_ast_types, columns=['token', 'node_type', 'parent_node_type'])
