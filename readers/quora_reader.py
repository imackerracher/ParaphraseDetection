from model.question_pairs import QuestionPairs

"""
The Quora Question Pairs corpus consists of a 2 questions,
and a flag specifying wether one of them is a duplicate.
It's questionable wether this is a suitable corpus:
a: What it is like to study in New Zealand for the Indian students?
b: What is it like to study in New Zealand as a foreign student?
These 2 questions are tagged as duplicate.
"""


def read(question_pairs_path):
    question_pairs = []
    with open(question_pairs_path, 'r') as f:
        for row in f.read().split('\n'):
            split_row = row.split('\t')
            #print(split_row)
            try:
                question_pairs.append(QuestionPairs(split_row[3], split_row[4], split_row[5]))
            except:
                #Small number of sentence pairs are missing information,
                #don't include them in corpus
                #print(split_row)
                continue
    return question_pairs