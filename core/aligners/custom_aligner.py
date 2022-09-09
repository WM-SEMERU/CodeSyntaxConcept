import re


class CustomAligner:

    def __init__(self, model_tokens, ast_tokens):
        self.model_tokens_queue = []
        self.ast_tokens = ast_tokens
        self.model_tokens = model_tokens
        self.last_ast_token_popped = None
        self.ast_tokens_last_match = None
        self.ast_token_popped_last_match = None
        self.ast_token_popped_concatenation = ''

    def align_tokens(self):
        self.last_ast_token_popped = self.ast_tokens.pop(0)
        for model_token in self.model_tokens:
            self.find_token_association(model_token)

    def find_token_association(self, model_token):
        # CHECK MATCH CONDITION
        # print('comparing model_token['+model_token+'] ast_token['+str(self.last_ast_token_popped)+']')
        if self.is_matching_token(model_token, self.last_ast_token_popped['token']):
            self.ast_tokens_last_match = self.ast_tokens.copy()
            self.ast_token_popped_last_match = self.last_ast_token_popped.copy()
            self.ast_token_popped_concatenation = ''
            self.model_tokens_queue.append({'token': model_token, 'association': self.last_ast_token_popped})
        elif len(self.ast_tokens) >= 1:
            self.ast_token_popped_concatenation += self.last_ast_token_popped['token'];
            self.last_ast_token_popped = self.ast_tokens.pop(0)
            self.find_token_association(model_token)
        else:
            self.ast_tokens = self.ast_tokens_last_match.copy()
            self.last_ast_token_popped = self.ast_token_popped_last_match.copy()
            self.model_tokens_queue.append({'token': model_token, 'association': None})

    def is_matching_token(self, model_token, ast_token):
        # print('comparing model_token['+model_token+'] ast_token['+str(self.last_ast_token_popped)+']')
        is_matching = False
        # if (model_token in ast_token):
        #  is_matching = True
        if (re.sub('[^A-Za-z0-9]+', '', model_token) in ast_token) and (
                len(re.sub('[^A-Za-z0-9]+', '', model_token)) >= 1):
            is_matching = True
        elif model_token in (self.ast_token_popped_concatenation + ast_token):
            is_matching = True
        return is_matching

    def get_model_tokens_queue(self):
        return self.model_tokens_queue
