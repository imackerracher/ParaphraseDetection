import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

    def lemmatize(self, text):
        l = WordNetLemmatizer()
        lemmas = [l.lemmatize(word) for word in text]
        return lemmas



def word_embeddings(dataset):
    """
    word2vec word embeddings
    :param question_pair:
    :return:
    """
    p = Preprocessor()
    X = []
    for i, question_pair in enumerate(dataset):
        a_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_a)))
        b_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_b)))

        a_vec = sum([model[word] for word in a_tokens if word in model])
        b_vec = sum([model[word] for word in b_tokens if word in model])

        try:
            sim = 1 - spatial.distance.cosine(a_vec, b_vec)
            X.append(sim)
        except:
            #print(a_vec)
            #print(b_vec)
            #print()
            X.append(0.5)
    return X

def token_overlap(dataset):
    p = Preprocessor()
    featureset = []
    for i, question_pair in enumerate(dataset):

        a_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_a)))
        b_tokens = set(p.stop_word_removal(p.lower_case(question_pair.question_b)))

        intersection_count = len(a_tokens.intersection(b_tokens))
        union_count = len(a_tokens.union(b_tokens))

        jaccard = 0 if intersection_count == 0 or union_count == 0 else (intersection_count / float(union_count))
        featureset.append(jaccard)

    return featureset



def create_lexicon(corpus):

    stop = set(stopwords.words('english'))
    lexicon = []
    p = Preprocessor()
    for qp in corpus:
        #qa_words = [word for word in nltk.word_tokenize(qp.question_a) if word not in stop and word not in punctuation]
        #qb_words = [word for word in nltk.word_tokenize(qp.question_b) if word not in stop and word not in punctuation]
        qa_words = p.lemmatize(p.punctuation_removal(p.lower_case(p.stop_word_removal(qp.question_a))))
        qb_words = p.lemmatize(p.punctuation_removal(p.lower_case(p.stop_word_removal(qp.question_b))))
        lexicon += qa_words
        lexicon += qb_words

    lexicon = nltk.FreqDist(w.lower() for w in lexicon)

    # print('peace was used: ', fd_doc_words['peace'])
    # print('america was used: ', fd_doc_words['america'])

    # Find the top x most frequently used words in the document
    top_words = lexicon.most_common(1000)

    # Return the top x most frequently used words
    return [t[0] for t in top_words]



def features_labels(data, lexicon):

    featureset = []
    p = Preprocessor()
    for qp in data:
        #qa = [lemmatizer.lemmatize(i) for i in nltk.word_tokenize(qp.question_a)]
        #qb = [lemmatizer.lemmatize(i) for i in nltk.word_tokenize(qp.question_b)]
        qa = p.lemmatize(p.punctuation_removal(p.lower_case(p.stop_word_removal(qp.question_a))))
        qb = p.lemmatize(p.punctuation_removal(p.lower_case(p.stop_word_removal(qp.question_b))))
        features = np.zeros(len(lexicon))
        for word in qa:
            if word.lower() in qb and word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1

        features = list(features)
        featureset.append(features)


    return featureset, labels




def labels(dataset):
    print('in labels')
    labels_ = []
    for q in dataset:
        if q.is_duplicate == '1':
            labels_.append([1,0])
        else:
            labels_.append([0,1])


    return labels_




def extract_features_and_labels(corpus):
    print('in extract features')
    #print(type(list(model['test'])))

    train_size = int(0.9 * len(corpus))

    train_x = [token_overlap(corpus[1:train_size]), word_embeddings(corpus[1:train_size])]
    test_x = [token_overlap(corpus[train_size:]), word_embeddings(corpus[train_size:])]

    train_y = labels(corpus[1:train_size])
    test_y = labels(corpus[train_size:])

    print(train_y)


    """print(len(a))
    print(len(b))

    for i in test_features:
        print(i)"""


    return train_x, train_y, test_x, test_y

