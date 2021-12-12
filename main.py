import os
import re
import string
import math

import numpy as np
import pandas


def get_data():
    data = pandas.read_csv('top10k.csv', sep=';')
    # data = pandas.read_csv('FinalStemmedSentimentAnalysisDataset.csv', sep=';')
    X = data['tweetText']
    y = data['sentimentLabel']
    # return X.values, y.values
    # return X.tolist(), y.tolist()
    return X, y

class naiveBayes():
    def __init__(self):

        self.tweet_num = {} # stores the amount of tweets that are 'positive' or 'negative'
        self.log_class_priors = {} # stores the prior probability of a message being 'positive' or 'negative'
        self.wc = {} # for each class, stores the amount of times a word appears
        self.vocab = set() # set of unique words.

    def item_count(self, text):
        wc = {}
        for w in text:
            wc[w] = wc.get(w, 0.0) + 1.0
        return wc

    def split(self, text):
        return re.split("\W+", text)

    def fit(self, X, y):

        # dataframe to npArray
        xArr = X.values
        yArr = y.values

        # total amount of tweets
        n = X.shape[0]

        # storing the amount of tweets that are 'positive' or 'negative'
        self.tweet_num['positive'] = np.count_nonzero(yArr)
        self.tweet_num['negative'] = n - self.tweet_num['positive']

        # calculating the prior probability of a message being 'positive' or 'negative'
        self.log_class_priors['negative'] = math.log(self.tweet_num['negative'])
        self.log_class_priors['positive'] = math.log(self.tweet_num['positive'])

        self.wc['negative'] = {}
        self.wc['positive'] = {}

        # iterating through all tweets
        for x, y in zip(xArr, yArr):

            c = 'negative'
            if y == 1:
                c = 'positive'

            # splitting tweet into dictionary word:count
            counts = self.item_count(self.split(x))

            # for each unique word in the tweet
            for word, count in counts.items():

                # creating new items should they not exist
                if word not in self.vocab:
                    self.vocab.add(word)
                if  word not in self.wc[c]:
                    self.wc[c][word] = 0.0

                # increasing the count for wrod in class
                self.wc[c][word] += count


        return n

def main():
    X, y = get_data()
    nb = naiveBayes()
    nb.fit(X, y)
    print("hola")
if __name__ == "__main__":
    main()