from sklearn import svm
import numpy as np
from nltk.corpus import stopwords
from definitions import project_root
from gensim.models import KeyedVectors
from scipy import spatial
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFECV

vecs = project_root + '/data/google_news_vectors/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(vecs, binary=True)
stop = set(stopwords.words('english'))

def word_embeddings(dataset):
    """
    word2vec word embeddings
    :param question_pair:
    :return:
    """
    X = np.zeros((len(dataset), 1))
    for i, question_pair in enumerate(dataset):
        a_tokens = set([token.lower() for token in question_pair.question_a if token not in stop])
        b_tokens = set([token.lower() for token in question_pair.question_b if token not in stop])

        a_vec = sum([model[word] for word in a_tokens if word in model])
        b_vec = sum([model[word] for word in b_tokens if word in model])

        try:
            sim = 1 - spatial.distance.cosine(a_vec, b_vec)
            X[i, 0] = sim
        except:
            #print(a_vec)
            #print(b_vec)
            #print()
            X[i, 0] = 0.5
    return X

def token_oerlap(dataset):
    X = np.zeros((len(dataset), 1))
    for i, question_pair in enumerate(dataset):
        a_tokens = set([token.lower() for token in question_pair.question_a if token not in stop])
        b_tokens = set([token.lower() for token in question_pair.question_b if token not in stop])

        intersection_count = len(a_tokens.intersection(b_tokens))
        union_count = len(a_tokens.union(b_tokens))

        jaccard = 0 if intersection_count == 0 or union_count == 0 else (intersection_count / float(union_count))
        X[i, 0] = jaccard

    return X


def extract_features(train, test):
    """
    :param train:
    :param test:
    :return:
    """
    train_features = np.concatenate([word_embeddings(train),
                                     token_oerlap(train)], axis=1)
    test_features = np.concatenate([word_embeddings(test),
                                    token_oerlap(test)], axis=1)

    return train_features, test_features

def split_corpus(corpus):
    """
    Split the corpus into train and test set
    :param corpus: The entire corpus
    :return: train_set (80%) and test_set (20%)
    """
    cut_off = int(len(corpus) * 0.8)
    train_set = corpus[:cut_off]
    test_set = corpus[cut_off:]

    return train_set, test_set

def evaluate(model, X, y):
    prediction = model.predict(X)

    acc = metrics.accuracy_score(y, prediction)
    print("Accuracy:", acc)
    return prediction

def classify(corpus):
    """
    :param corpus: Only quora question corpus atm
    :return:
    """

    train, test = split_corpus(corpus)

    X_train, X_test = extract_features(train, test)

    y_train = [qp.is_duplicate for qp in train]
    y_test = [qp.is_duplicate for qp in test]

    svm_model = svm.LinearSVC()
    svm_model.fit(X_train, y_train)
    prediction = evaluate(svm_model, X_test, y_test)

    print('\n'*7)
    print(y_test)

    """for i, qp in enumerate(test):
        print('actual: ', y_test[i])
        print('features: ', X_test[i])
        print('predicted: ', prediction[i])
        print(qp.question_a)
        print(qp.question_b)
        print()

    print()

    p = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for i in range(len(prediction)):
        if prediction[i] == '1':
            if y_test[i] == '1':
                p['tp'] += 1
            elif y_test[i] == '0':
                p['fp'] += 1
        elif prediction[i] == '0':
            if y_test[i] == '0':
                p['tn'] += 1
            elif y_test[i] == '1':
                p['fn'] += 1
    print('predicions: ', p)"""








