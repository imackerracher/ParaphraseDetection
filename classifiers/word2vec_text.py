from gensim.models import KeyedVectors
import time



def load(data_folder):
    vecs = data_folder + '/google_news_vectors/GoogleNews-vectors-negative300.bin'

    a = time.time()
    model = KeyedVectors.load_word2vec_format(vecs, binary=True)
    print(time.time()-a)
    print(model['word'])
    print(time.time()-a)