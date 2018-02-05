"""
Structure used for msrp_distribute and opinosis_distribute corpora.

Idea: make pairs out of the groups. I.e. if there is a group of 3
paraphrases: a, b, c then this would result in:
a b
a c
b c
Would be more in line with the other corpora. Especially Quora Question Pairs.
"""

class ParaphraseGroup:

    def __init__(self, sentence_list):
        self.sentence_list = sentence_list