from definitions import project_root
#Readers
from readers.sick_reader import read as read_sick_corpus
from readers.msrp_opinosis_reader import read as read_msrp_distribute_corpus
from readers.quora_reader import read as read_quora_corpus
from readers.webis_reader import read as read_webis_corpus
from readers.sts_reader import read as read_sts_corpus
from model.phrase_pair import PhrasePair
from random import shuffle
#Classifiers
#from classifiers.log_linear import classify as log_linear_classify
#from classifiers.svm import classify as svm_classify
#from classifiers.ffnn import classify as ffnn_classify
#from classifiers.ffnn import dump
#from classifiers.nn_featuresets import extract_features_and_labels

import scipy.special


data_folder = project_root + '/data'

def sick():
    sick_corpus_path = data_folder + '/sick/SICK.txt'
    sick_corpus = read_sick_corpus(sick_corpus_path)
    scores = []
    for s in sick_corpus:
        print(s.sentence_A)
        print(s.sentence_B)
        print(s.relatedness_score)
        scores.append(float(s.relatedness_score))
        print(s.entailment_label)
        print()
    scores = list(set(scores))
    for i in sorted(scores):
        print(i)
    print(len(sick_corpus))
    return sick_corpus

def msrp_distribute():
    msrp_distribute_corpus_path = data_folder + '/msrp_distribute'
    phrases_path = msrp_distribute_corpus_path + '/phrases.txt'
    phrase_groups_path = msrp_distribute_corpus_path + '/phrase_groups.csv'
    msrp_distribute_corpus = read_msrp_distribute_corpus(phrases_path, phrase_groups_path)
    """for i in msrp_distribute_corpus:
        for p in i.sentence_list:
            print(p)
        print()
    print(len(msrp_distribute_corpus))"""
    return msrp_distribute_corpus

def opinosis_distribute():
    opinosis_distribute_corpus_path = data_folder + '/opinosis_distribute'
    phrases_path = opinosis_distribute_corpus_path + '/phrases.txt'
    phrase_groups_path = opinosis_distribute_corpus_path + '/phrase_groups.csv'
    opinosis_distribute_corpus = read_msrp_distribute_corpus(phrases_path, phrase_groups_path)
    """for i in opinosis_distribute_corpus:
        for p in i.sentence_list:
            print(p)
        print()
    print(len(opinosis_distribute_corpus))"""
    return opinosis_distribute_corpus

def quora():
    quora_corpus_path = data_folder + '/quora/quora_question_pairs.txt'
    quora_questions_corpus = read_quora_corpus(quora_corpus_path)
    """for pair in quora_questions_corpus:
        print(pair.question_a)
        print(pair.question_b)
        print(pair.is_duplicate)
        print()"""
    return quora_questions_corpus

def webis():
    """
    Not sentence pairs -> not ideal, maybe for testing?
    :return:
    """
    webis_corpus_path = data_folder + '/Webis-CPC-11'
    webis_corpus = read_webis_corpus(webis_corpus_path)
    for w in webis_corpus[10:]:
        print(w.paraphrase)
        print('#'*100)
        print(w.original)
        print(w.is_paraphrase)
        print('\n\n\n\n###########################\n\n\n\n')

    print(len(webis_corpus))
    return webis_corpus

def sts():
    """
    Consists of several corpora
    Similarity score between 0 and 1
    Kind of weird grading
    """
    sts_2012_test = data_folder + '/STS/STS2012-test'
    sts_2012_train = data_folder + '/STS/STS2012-train'
    sts_2013_test = data_folder + '/STS/STS2013-test'
    sts_2013_trial = data_folder + '/STS/STS2013-trial'
    sts_2014_trial = data_folder + '/STS/STS2014-trial'
    folders = [sts_2012_test, sts_2012_train, sts_2013_test, sts_2013_trial, sts_2014_trial]
    sts_corpus = []
    for path in folders:
        sts_corpus += read_sts_corpus(path)

    scores = {}
    for pair in sts_corpus:
        if pair.similarity_score < 1.0:
            """print(pair.sentence_a)
            print(pair.sentence_b)
            print(pair.similarity_score)
            print()"""
            if str(pair.similarity_score) not in scores:
                scores[str(pair.similarity_score)] = pair

    for k in scores:
        p = scores[k]
        print(p.sentence_a)
        print(p.sentence_b)
        print(p.similarity_score)
        print()

    print(len(sts_corpus))

    return sts_corpus



def concatenate_corpora():
    corpus = []



    """quora_corpus = quora()[1:]
    for qp in quora_corpus:
        phrase_pair = PhrasePair(qp.question_a, qp.question_b, [1,0] if qp.is_duplicate == '1' else [0,1], 'quora')"""

    """msrp_corpus = msrp_distribute()
    for sents in msrp_corpus:
        sl = sents.sentence_list
        print(len(sl))
        for i in sl:
            print(i)
        print()
        for i in range(len(sl)):
            for j in range(i+1, len(sl[1:]) + 1):
                corpus.append(PhrasePair(sl[i], sl[j], [1,0], 'msrp_distribute'))"""

    """opinosis_corpus = opinosis_distribute()
    for sents in opinosis_corpus:
        sl = sents.sentence_list
        print(len(sl))
        for i in sl:
            print(i)
        print()
        for i in range(len(sl)):
            for j in range(i + 1, len(sl[1:]) + 1):
                corpus.append(PhrasePair(sl[i], sl[j], [1, 0], 'opinosis_distribute'))"""






    """for i in corpus:
        print(i.phrase_a)
        print(i.phrase_b)
        print()
    print(len(corpus))"""



if __name__ == '__main__':
    #sick = len(sick())
    #msrp = len(msrp_distribute())
    #opinosis = len(opinosis_distribute())
    #quora = len(quora())
    #webis = len(webis())
    sts = len(sts())

    """print('sick: ', sick)
    print('msrp: ', msrp)
    print('opinosis: ', opinosis)
    print('quora: ', quora)
    print('webis: ', webis)
    print('sts: ', sts)

    print()
    print('total: ', sick + msrp + opinosis + quora + webis + sts)"""


    """corpus = quora()
    corpus = corpus
    count = {}
    for q in corpus:
        if q.is_duplicate in count:
            count[q.is_duplicate] += 1
        else:
            count[q.is_duplicate] = 1

    print(count)"""



    #extract_features_and_labels(corpus)
    #dump(corpus)

    #concatenate_corpora()

    """x = 0
    for i in opinosis_distribute():
        x += scipy.special.binom(len(i.sentence_list), 2)

    print(x)"""






