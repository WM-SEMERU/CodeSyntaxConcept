import numpy as np

'''DEPRECATED'''
class NeedlemanWunch:
    # scores
    MATCH_SCORE = 1
    MISMATCH_SCORE = -1
    GAP_SCORE = -2

    # constructor
    def __init__(self, model_tokens, ast_tokens) -> None:
        self.model_tokens = model_tokens
        self.ast_tokens = ast_tokens
        self.model_tokens_queue = []

        self.ast_tokens.insert(0, None)
        self.model_tokens.insert(0, None)
        self.score_matrix = np.zeros(shape=(len(model_tokens), len(ast_tokens)))

    def find_optimal_sequence(self):
        # initialize matrix rows with gap penalty
        for row_idx in range(1, len(self.model_tokens)):
            self.score_matrix[row_idx, 0] = self.score_matrix[row_idx - 1, 0] + NeedlemanWunch.GAP_SCORE
        # initialize matrix columns with gap penalty
        for column_idx in range(1, len(self.ast_tokens)):
            self.score_matrix[0, column_idx] = self.score_matrix[0, column_idx - 1] + NeedlemanWunch.GAP_SCORE
        # calculate scores and update matrix
        for row_idx in range(1, len(self.model_tokens)):
            for column_idx in range(1, len(self.ast_tokens)):
                self.score_matrix[row_idx, column_idx] = self.calculate_score(row_idx, column_idx)
        # find optimal path
        score_matrix_rows, score_matrix_columns = self.score_matrix.shape
        self.find_optimal_path(score_matrix_rows - 1, score_matrix_columns - 1)
        # clean row values
        for token_idx in range(0, len(self.model_tokens_queue)):
            self.model_tokens_queue[token_idx] = {'token': self.model_tokens_queue[token_idx]['token'],
                                                  'association': self.model_tokens_queue[token_idx]['association']}
        self.model_tokens_queue = self.model_tokens_queue[::-1]

    def calculate_score(self, row_idx, column_idx):
        # vertical
        is_matching_position = self.model_tokens[row_idx - 1] == self.ast_tokens[column_idx - 1]
        # match/mismatch
        diagonal_value = self.score_matrix[row_idx - 1, column_idx - 1] + (
            NeedlemanWunch.MATCH_SCORE if is_matching_position else NeedlemanWunch.MISMATCH_SCORE)
        # horizontal gap (moving across vertical)
        horizontal_gap_value = self.score_matrix[row_idx - 1, column_idx] + NeedlemanWunch.GAP_SCORE
        # vertical gap (moving across horizontal)
        vertical_gap_value = self.score_matrix[row_idx, column_idx - 1] + NeedlemanWunch.GAP_SCORE
        return max(diagonal_value, horizontal_gap_value, vertical_gap_value)

    def find_optimal_path(self, row_idx, column_idx):
        if row_idx <= 1 and column_idx <= 1:
            return None
        next_row_idx = row_idx
        next_column_idx = column_idx
        # starting from the last position
        score_matrix_rows, score_matrix_columns = self.score_matrix.shape
        if (score_matrix_rows - 1) == row_idx and (score_matrix_columns - 1) == column_idx:
            self.model_tokens_queue.append(
                {'row': row_idx, 'token': self.model_tokens[row_idx], 'association': self.ast_tokens[column_idx]})
        # assest best route
        diagonal_value = self.score_matrix[row_idx - 1, column_idx - 1]
        horizontal_gap_value = self.score_matrix[row_idx - 1, column_idx]
        vertical_gap_value = self.score_matrix[row_idx, column_idx - 1]
        # max_score
        max_score = max(diagonal_value, horizontal_gap_value, vertical_gap_value)
        # match/mismatch score
        if diagonal_value == max_score:
            self.model_tokens_queue.append({'row': row_idx - 1, 'token': self.model_tokens[row_idx - 1],
                                            'association': self.ast_tokens[column_idx - 1]})
            next_row_idx -= 1
            next_column_idx -= 1
        # horizontal gap (moving across vertical)
        if horizontal_gap_value == max_score:
            self.model_tokens_queue.append({'row': row_idx - 1, 'token': self.model_tokens[row_idx - 1],
                                            'association': self.ast_tokens[column_idx]})
            next_row_idx -= 1
        # vertical gap (moving across horizontal) -> replace
        if vertical_gap_value == max_score:
            next_column_idx -= 1
            token_queue_idx = [token_idx for token_idx in range(0, len(self.model_tokens_queue)) if
                               self.model_tokens_queue[token_idx]['row'] == row_idx][0]
            self.model_tokens_queue[token_queue_idx] = {'row': row_idx, 'token': self.model_tokens[row_idx],
                                                        'association': self.ast_tokens[column_idx - 1]}
        self.find_optimal_path(next_row_idx, next_column_idx)

    def get_model_tokens_queue(self):
        return self.model_tokens_queue
