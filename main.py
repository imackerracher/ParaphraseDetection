from definitions import project_root
#Readers
from readers.sick_reader import read as read_sick_corpus
from readers.msrp_opinosis_reader import read as read_msrp_distribute_corpus
from readers.quora_reader import read as read_quora_corpus
from readers.webis_reader import read as read_webis_corpus
from readers.sts_reader import read as read_sts_corpus
from random import shuffle
#Classifiers
#from classifiers.log_linear import classify as log_linear_classify
from classifiers.svm import classify as svm_classify


data_folder = project_root + '/data'

def sick():
    sick_corpus_path = data_folder + '/sick/SICK.txt'
    sick_corpus = read_sick_corpus(sick_corpus_path)
    """for s in sick_corpus:
        print(s.sentence_A)
        print(s.sentence_B)
        print(s.relatedness_score)
        print(s.entailment_label)
        print()"""
    return sick_corpus

def msrp_distribute():
    msrp_distribute_corpus_path = data_folder + '/msrp_distribute'
    phrases_path = msrp_distribute_corpus_path + '/phrases.txt'
    phrase_groups_path = msrp_distribute_corpus_path + '/phrase_groups.csv'
    msrp_distribute_corpus = read_msrp_distribute_corpus(phrases_path, phrase_groups_path)
    """for i in msrp_distribute_corpus:
        for p in i.sentence_list:
            print(p)
        print()"""
    return msrp_distribute_corpus

def opinosis_distribute():
    opinosis_distribute_corpus_path = data_folder + '/opinosis_distribute'
    phrases_path = opinosis_distribute_corpus_path + '/phrases.txt'
    phrase_groups_path = opinosis_distribute_corpus_path + '/phrase_groups.csv'
    opinosis_distribute_corpus = read_msrp_distribute_corpus(phrases_path, phrase_groups_path)
    """for i in msrp_distribute_corpus:
        for p in i.sentence_list:
            print(p)
        print()"""
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
    webis_corpus_path = data_folder + '/Webis-CPC-11'
    webis_corpus = read_webis_corpus(webis_corpus_path)
    """for w in webis_corpus[:10]:
        print(w.id)
        print(w.paraphrase)
        print(w.original)
        print(w.is_paraphrase)
        print('\n\n\n\n###########################\n\n\n\n')"""
    return webis_corpus

def sts():
    """
    Consists of several corpora
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

    #print(len(sts_corpus))
    """for pair in sts_corpus:
        if pair.similarity_score < 1.0:
            print(pair.sentence_a)
            print(pair.sentence_b)
            print(pair.similarity_score)
            print()"""

    return sts_corpus

#test

if __name__ == '__main__':
    """sick = len(sick())
    msrp = len(msrp_distribute())
    opinosis = len(opinosis_distribute())
    quora = len(quora())
    webis = len(webis())
    sts = len(sts())

    print('sick: ', sick)
    print('msrp: ', msrp)
    print('opinosis: ', opinosis)
    print('quora: ', quora)
    print('webis: ', webis)
    print('sts: ', sts)

    print()
    print('total: ', sick + msrp + opinosis + quora + webis + sts)"""

    quora_0 = []
    quora_1 = []
    i = 0
    corpus = quora()
    """length = 100000
    while len(quora_0) < length or len(quora_1) < length:
        q = corpus[i]
        if q.is_duplicate == '0' and len(quora_0) < length:
            quora_0.append(q)
        elif q.is_duplicate == '1' and len(quora_1) < length:
            quora_1.append(q)
        i += 1

    quora = quora_0 + quora_1
    shuffle(quora)

    count = {}
    for q in quora:
        if q.is_duplicate in count:
            count[q.is_duplicate] += 1
        else:
            count[q.is_duplicate] = 1


    print(count)
    print()"""
    #log_linear_classify(corpus[:500])

    count = {}
    for q in corpus:
        if q.is_duplicate in count:
            count[q.is_duplicate] += 1
        else:
            count[q.is_duplicate] = 1

    print(count)
    print()


    shuffle(corpus)
    svm_classify(corpus)


