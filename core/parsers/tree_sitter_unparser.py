from tree_sitter import Language, Parser

LANGUAGES = [
    'tree-sitter-python'
]

Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',
    # Include one or more languages
    LANGUAGES
)


class TreeSitterParser:
    PY_LANGUAGE = Language('build/my-languages.so', 'python')

    def __init__(self, source_code, language: Language) -> None:
        self.source_code = source_code
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
