import numpy as np
import nltk
from nltk.corpus import stopwords
from definitions import project_root
from gensim.models import KeyedVectors
from scipy import spatial
from string import punctuation


vecs = project_root + '/data/google_news_vectors/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(vecs, binary=True)
stop = set(stopwords.words('english'))

class Preprocessor:

    def lower_case(self, text):
        return [word.lower() for word in text]

    def stop_word_removal(self, text):
        return [word for word in text if word not in stop]

    def punctuation_removal(self, text):
        return [word for word in text if word not in punctuation]

    def stem(self, text, stemmer='porter'):
        if stemmer == "porter":
            stemmer_ = nltk.PorterStemmer()
        elif stemmer == "lancaster":
            stemmer_ = nltk.LancasterStemmer()
        elif stemmer == "snowball":
            stemmer_ = nltk.SnowballStemmer('english')
        else:
            raise ValueError("Unknown stemmer: %s" % stemmer)

        stemmed_words = [stemmer_.stem(word) for word in text]

        return stemmed_words



def word_embeddings(dataset):
    """
    word2vec word embeddings
    :param question_pair:
    :return:
    """
    p = Preprocessor()
    X = np.zeros((len(dataset), 1))
    for i, question_pair in enumerate(dataset):
        #a_tokens = set([token.lower() for token in question_pair.question_a if token not in stop])
        #b_tokens = set([token.lower() for token in question_pair.question_b if token not in stop])

        a_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_a)))
        b_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_b)))

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
    p = Preprocessor()
    X = np.zeros((len(dataset), 1))
    for i, question_pair in enumerate(dataset):
        #a_tokens = set([token.lower() for token in question_pair.question_a if token not in stop])
        #b_tokens = set([token.lower() for token in question_pair.question_b if token not in stop])

        a_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_a)))
        b_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_b)))

        intersection_count = len(a_tokens.intersection(b_tokens))
        union_count = len(a_tokens.union(b_tokens))

        jaccard = 0 if intersection_count == 0 or union_count == 0 else (intersection_count / float(union_count))
        X[i, 0] = jaccard

    return X


def construct_lexicon(dataset, n_most_common):
    """
    :param dataset:
    :param n_most_common:
    :return:
    """
    p = Preprocessor()
    stemmer_ = 'porter'
    tokens_ = []
    for question_pair in dataset:
        tokens_a = p.stem(p.stop_word_removal(p.lower_case(question_pair.question_a)), stemmer=stemmer_)
        tokens_b = p.stem(p.stop_word_removal(p.lower_case(question_pair.question_b)), stemmer=stemmer_)
        tokens_.append(tokens_a)
        tokens_.append(tokens_b)

    fd_doc_stems = nltk.FreqDist(s for s in tokens_)

    # Find the top x stems
    lexicon = fd_doc_stems.most_common(n_most_common)

    return lexicon


def tokens(dataset, n_most_common):
    """
    NOT CONSISTENT WITH OTHER FEATURE STRUCTURES

    Construct a dictionary with the n most common words (excluding stop words). Each token is then represented
    as a sparse vector with n-1 zeros and 1 one. A text passage is represented by as many ones as there are
    distinct tokens from the lexicon represented in the text passage. Simple bag of words model.
    :param dataset:
    :return:
    """

    lexicon = construct_lexicon(dataset, n_most_common)
    p = Preprocessor()
    X = np.zeros(len(dataset), 1)
    def sparse_vector(text):
        vec = np.zeros(len(lexicon))
        current_words = p.stem(p.stop_word_removal(p.lower_case(text)))
        for word in current_words:
            index_value = lexicon.index(word)
            vec[index_value] += 1

        return vec

    for i, question_pair in enumerate(dataset):
        a_vec = sparse_vector(question_pair.question_a)
        b_vec = sparse_vector(question_pair.question_b)

        try:
            sim = 1 - spatial.distance.cosine(a_vec, b_vec)
            X[i, 0] = sim
        except:
            print('Exception')
            X[i, 0] = 0.5

    return X




"""
TODO Features:
- stem/lemmatize words before overlap/word embeddings/sparse vectors ...
    - have this as an option
- TFIDF
- experiment with different word embeddings relations
- Wikidata
    - NE algorithms
- Wordnet
"""




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