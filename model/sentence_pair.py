"""
Structure used for SICK corpus
"""

class SentencePair:
    def __init__(self, pair_ID, sentence_A, sentence_B, entailment_label, relatedness_score, entailment_AB, entailment_BA,
                 sentence_A_original, sentence_B_original, sentence_A_dataset, sentence_B_dataset, SemEval_set):
        self.pair_ID = pair_ID
        self.sentence_A = sentence_A
        self.sentence_B = sentence_B
        self.entailment_label = entailment_label
        self.relatedness_score = relatedness_score
        self.entailment_AB = entailment_AB
        self.entailment_BA = entailment_BA
        self.sentence_A_original = sentence_A_original
        self.sentence_B_original = sentence_B_original
        self.sentence_A_dataset = sentence_A_dataset
        self.sentence_B_dataset = sentence_B_dataset
        self.SemEval_set = SemEval_set
