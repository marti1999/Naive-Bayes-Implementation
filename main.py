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

        self.tweet_num = {}  # stores the amount of tweets that are 'positive' or 'negative'
        self.log_prior_probability = {}  # stores the log prior probability of a message being 'positive' or 'negative'
        self.wc = {}  # for each class, stores the amount of times a word appears
        self.dictionary = set()  # set of unique words.

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
        self.log_prior_probability['negative'] = math.log(self.tweet_num['negative'])
        self.log_prior_probability['positive'] = math.log(self.tweet_num['positive'])

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
                if word not in self.dictionary:
                    self.dictionary.add(word)
                if word not in self.wc[c]:
                    self.wc[c][word] = 0.0

                # increasing the count for word in class
                self.wc[c][word] += count

    def predict(self, X):
        xArr = X.values()
        result = []

        for tweet in xArr:
            word_counts = self.item_count(self.split(tweet))
            positive_count = 0
            negative_count = 0
            for word, _ in word_counts.items():
                if word not in self.dictionary:
                    continue

                # Applying Naive Bayes
                # We need to calculate p(w_i | positive) and p(w_i | negative)
                # The numerator is how many times w_i appears in a tweet of such class, divided by the count of
                # all words in the tweets of the class.
                # Since we can't calculate log of 0, we use Laplace Smoothing, adding 1 to the numerator.
                # In order to balance it, the size of the dictionary must be added to the numerator
                log_positive = math.log((self.wc['positive'].get(word, 0.0) + 1)
                                        / (self.tweet_num['positive'] + len(self.dictionary)))
                log_negative = math.log((self.wc['negative'].get(word, 0.0) + 1)
                                        / (self.tweet_num['negative'] + len(self.dictionary)))

                positive_count += log_positive
                negative_count += log_negative

            # Adding prior probability of each class
            positive_count += self.log_prior_probability['positive']
            negative_count += self.log_prior_probability['negative']

            if positive_count > negative_count:
                result.append(1)
            else:
                result.append(0)

        return result


def main():
    X, y = get_data()
    nb = naiveBayes()
    nb.fit(X, y)
    print("hola")


if __name__ == "__main__":
    main()
