from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import random

random.seed(123)

def delete_character(word):
    ''' randomly remove one character of a word '''
    i = random.randint(0, len(word) - 1)
    noisy_word = word[0:i] + word[i + 1:]
    return noisy_word

class Dataset(object):
    ''' defines a dataset that returns a dictionnary with words and its noisy
    versions - currently deleting a certain number of characters
    The idea is to build a mispelling corrector using a RNN.
    '''
    def __init__(self, subset, n_character_deleted=1):
        assert subset in ['train', 'valid', 'test']
        twenty_news_groups = fetch_20newsgroups(subset=subset)
        count_vect = CountVectorizer()
        count_vect.fit(twenty_news_groups.data)
        self.words = count_vect.vocabulary_.keys()
        random.shuffle(self.words)
        self.idx = 0

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= len(self.words):
            self.idx = 0
            raise StopIteration
        else:
            word_target = self.words[self.idx]
            word_source = delete_character(word_target)
            self.idx += 1
            return word_source, word_target
