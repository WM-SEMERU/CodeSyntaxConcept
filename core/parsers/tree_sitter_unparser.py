import json
import os

from tree_sitter import Language, Parser

LANGUAGES = [
    'tree-sitter-go',
    'tree-sitter-javascript',
    'tree-sitter-python'
]

Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',
    # Include one or more languages
    LANGUAGES
)

GO_LANGUAGE = Language('build/my-languages.so', 'go')
JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
PY_LANGUAGE = Language('build/my-languages.so', 'python')

all_node_types = {
    p: json.load(open(os.path.join(p, "core", "node-types.json"), "r")) for p in LANGUAGES
}


def get_lang_types(all_node_types, lang):
    node_types = [node_type["type"] for node_type in all_node_types[lang]]
    node_subtypes = [
        node_subtype["type"]
        for node_type in all_node_types["tree-sitter-python"]
        if "subtypes" in node_type
        for node_subtype in node_type["subtypes"]
    ]
    return list(set(node_types + node_subtypes))


python_node_types = get_lang_types(all_node_types, "tree-sitter-python")


class TreeSitterParser:

    def __init__(self, source_code, language) -> None:
        self.nodes = []
        self.tokens = []
        self.parser = Parser()
        self.parser.set_language(language)

    def tokenize(self):
        tree = self.parser.parse(bytes(self.source_code, "utf8"))
        self.traverse(tree.root_node)
        self.retrieve_tokens()

    def traverse(self, node) -> None:
        if node.type == 'string':
            self.nodes.append(node)
            return
        for child_node in node.children:
            self.traverse(child_node)
        if not node.children:
            self.nodes.append(node)

    def retrieve_tokens(self):
        for node in self.nodes:
            self.tokens.append({'token': node.text.decode("utf-8"), 'family': node.type})

    def get_tokens(self):
        return self.tokens
