from model.sentence_pair import SentencePair

"""
The SICK dataset is built up as follows for each line:
pair_ID
sentence_A
sentence_B
entailment_label
relatedness_score
entailment_AB
entailment_BA
sentence_A_original
sentence_B_original
sentence_A_dataset
sentence_B_dataset
SemEval_set
"""


def read(full_file_path):
    f = open(full_file_path, 'r')
    file = f.read()
    lines = file.split('\n')

    sick_corpus = []
    for line in lines[1:]:
        line = line.split('\t')
        try:
            new_sentence_pair = SentencePair(line[0], line[1], line[2], line[3], line[4], line[5],
                                             line[6], line[7], line[8], line[9], line[10], line[11])
            sick_corpus.append(new_sentence_pair)
        except Exception as e:
            continue

    return sick_corpus
