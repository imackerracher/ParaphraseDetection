from model.sts_pair import STSPair
import os


"""
STS corpus: each sentence pair has a similarity score (float between 0 and 1).
Probably should only take sentence pairs that exceed certain threshold.
7640 sentence pairs
"""

def read_file(path):
    with open(path, 'r') as f:
        file = f.read().split('\n')
    return file


def read(sts_corpus_path):
    file_names = [n for n in os.listdir(sts_corpus_path) if n.endswith('.txt')]
    middle = int(len(file_names)/2)
    score_files = [s for fn in file_names[:middle] for s in read_file(sts_corpus_path + '/' + fn)]
    txt_files = [t for fn in file_names[middle:] for t in read_file(sts_corpus_path + '/' + fn)]
    file_pairs = zip(score_files, txt_files)

    def new_sts_pair(tup):
        try:
            score = tup[0]
            texts = tup[1].split('\t')
            return STSPair(texts[0], texts[1], float(score))
        except:
            return None

    sts_corpus = filter(lambda x: x is not None, map(new_sts_pair, file_pairs))

    return sts_corpus







