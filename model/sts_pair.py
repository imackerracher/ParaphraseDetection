"""
This structure is used to read in the different STS corpora
"""

class STSPair:

    def __init__(self, sentence_a, sentence_b, similarity_score):
        self.sentence_a = sentence_a
        self.sentence_b = sentence_b
        self.similarity_score = similarity_score