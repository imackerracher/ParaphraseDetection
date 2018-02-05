import random
"""
Used for Quora Question Pairs corpus
"""

class QuestionPairs:

    def __init__(self, question_a, question_b, is_duplicate):
        self.question_a = question_a
        self.question_b = question_b
        #for testing
        #self.is_duplicate = str(random.choice([i for i in range(10)]))
        self.is_duplicate = is_duplicate
        #self.is_duplicate = '10' if is_duplicate == '0' else '20'