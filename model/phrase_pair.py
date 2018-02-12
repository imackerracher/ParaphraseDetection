"""
Concatenation of several corpora
"""


class PhrasePair:

    def __init__(self, phrase_a, phrase_b, label, origin):
        self.phrase_a = phrase_a
        self.phrase_b = phrase_b
        self.label = label
        self.origin = origin

