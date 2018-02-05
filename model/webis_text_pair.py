"""
Used for the Webis corpus
"""

class WebisTextPair:

    def __init__(self, id, paraphrase, original, is_paraphrase):
        self.id = id
        self.paraphrase = paraphrase
        self.original = original
        self.is_paraphrase = is_paraphrase