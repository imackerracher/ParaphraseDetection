from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors
import numpy as np
from scipy import spatial
from nltk import word_tokenize
from nltk.corpus import stopwords
from definitions import project_root
import random

vecs = project_root + '/data/google_news_vectors/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(vecs, binary=True)

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


def token_overlap(question_pair, features):
    """
    Use the jaccard coefficient for token overlap as a feature
    :param question_pair: Current question pair for which the feature is added
    :param features: The features that have been added
    :return: updated features
    """

    a_tokens = set([token.lower() for token in question_pair.question_a])
    b_tokens = set([token.lower() for token in question_pair.question_b])

    intersection_count = len(a_tokens.intersection(b_tokens))
    union_count = len(a_tokens.union(b_tokens))

    jaccard = 0 if intersection_count == 0 or union_count == 0 else (intersection_count/float(union_count))
    features['token_overlap'] = jaccard
    return features


def word_embeddings(question_pair, features):
    """
    word2vec word embeddings
    :param question_pair:
    :return:
    """
    stop = set(stopwords.words('english'))
    a_tokens = set([token.lower() for token in question_pair.question_a if token not in stop])
    b_tokens = set([token.lower() for token in question_pair.question_b if token not in stop])

    a_vec = sum([model[word] for word in a_tokens if word in model])
    b_vec = sum([model[word] for word in b_tokens if word in model])


    sim = 1- spatial.distance.cosine(a_vec, b_vec)
    features['word_embedding_distance'] = sim
    return features


def sentence_length_dif(question_pair, features):
    """
    Sentence as the feature
    :param question_pair:
    :param features:
    :return:
    """
    a_length = len([token for token in question_pair.question_a])
    b_length = len([token for token in question_pair.question_b])

    dif = a_length - b_length
    features['sentence_length_difference'] = dif
    return features

def random_choice(features):
    """
    to test random choice
    :param question_pari:
    :param features:
    :return:
    """
    features['random'] = random.choice([0,1])
    return features

def test(features):
    """
    For test purpose
    :param features:
    :return:
    """
    features['test'] = 0
    return features



def extract_features(set):
    """
    This method extracts the features that are used to train the model and returns them as feature_dictionaries.
    :param set: train_set or test_set
    :return: List of feature dictionaries
    """
    feature_dicts = []
    labels = []
    for question_pair in set:
        features = {}
        #features = token_overlap(question_pair, features)
        #features = word_embeddings(question_pair, features)
        #features = word_embeddings(question_pair, features)
        #features = random_choice(features)
        features = test(features)
        label = question_pair.is_duplicate
        feature_dicts.append(features)
        labels.append(label)

    return feature_dicts, labels

def get_feature_matrix(feature_dicts):
    """
    Transform the feature dictionaries into vector, and feature matrix
    :param feature_dicts:
    :return:
    """
    vector = DictVectorizer()

    feature_matrix = vector.fit_transform(feature_dicts)

    return feature_matrix, vector

def logistic_regression(train_feature_matrix, train_labels, c_value=1.0):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', verbose=10, C=c_value)

    # Fit the model
    model.fit(train_feature_matrix, train_labels)

    # Return the model
    return model



def classify(corpus):
    """
    :param corpus: Implemented at the moment only for Quora Duplicate Questions Corpus
    :return: A trained classiier that predict classifies 2 given questions as being duplicates
    """

    train_set, test_set = split_corpus(corpus)
    train_feature_dicts, train_labels = extract_features(train_set)
    train_feature_matrix, train_feature_transform_vector = get_feature_matrix(train_feature_dicts)

    model = logistic_regression(train_feature_matrix, train_labels, c_value=1.0)
    train_predict = model.predict(train_feature_matrix)
    for i in range(len(train_predict)):
        print(train_predict[i], train_labels[i])


    """for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 3, 5, 7, 9]:
        model = logistic_regression(train_feature_matrix, train_labels, c_value=i)
        train_predict = model.predict(train_feature_matrix)
        train_acc = np.mean(train_predict == train_labels)
        print("Train set accuracy: %.4f" % train_acc, ' with C value of ', i)
        count = {'1': 0, '0': 0}
        for l in train_predict:
            if l in count: count[l] += 1
        print(count)
        print(len(train_predict))
        print()"""





