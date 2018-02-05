from model.paraphrase_group import ParaphraseGroup
import csv

"""
For the MSRP-Distribute and opinosis_distribute corpora 2 files are needed (each).
phrases.txt contains all the sentences.
phrase_groups.csv contains the index of all the sentences
belonging to each group:
phrase_index,paraphrase_group_index
    0,0
    1,0
    2,0
    3,1
    4,1
    5,1

That says that the sentences on lines 0,1,and 2 have the same meaning,
"""

def get_phrase_groups(phrase_groups_path):
    phrase_groups = {}
    with open(phrase_groups_path, 'r') as f:
        reader = csv.reader(f)
        #rows = [row for row in reader]
        for row in reader:
            if row[1] in phrase_groups:
                phrase_groups[row[1]].append(row[0])
            else:
                phrase_groups[row[1]] = [row[0]]

    return phrase_groups


def get_phrases(phrases_path):
    with open(phrases_path, 'r') as f:
        phrases = f.read().split('\n')
    return phrases[:-1]


def read(phrases_path, phrase_groups_path):
    phrase_groups = get_phrase_groups(phrase_groups_path)
    phrases = get_phrases(phrases_path)
    paraphrase_groups = []
    for k in phrase_groups:
        #for i in phrase_groups[k]:
        #    print(type(i))
        #print()
        try:
            paraphrase_groups.append(ParaphraseGroup([phrases[int(index)] for index in phrase_groups[k]]))
        except Exception as e:
            continue
    return paraphrase_groups

