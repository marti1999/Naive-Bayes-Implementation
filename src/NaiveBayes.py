import math
import sys
from collections import Counter

import numpy as np
import sklearn


# Inherit from sklearn.base.BaseEstimator
# https://scikit-learn.org/stable/developers/develop.html
# Needed in order to use some sklearn modules like cross_validate
#   "All estimators in the main scikit-learn codebase should inherit from sklearn.base.BaseEstimator."
class naiveBayes(sklearn.base.BaseEstimator):
    def __init__(self, laplace_smoothing=1, dictionary_size=sys.maxsize):

        self.laplace_smoothing = laplace_smoothing
        self.log_prior_probability = {}  # stores the log prior probability of a message being 'positive' or 'negative'
        self.wc = {}  # for each class, stores the amount of times a word appears
        self.dictionary = set()  # set of unique words.
        self.dictionary_size = dictionary_size # maximum dictionary size

    # function used to test different dictionary sizes.
    # as it is not used inside the class, it is static
    @staticmethod
    def count_total_words(X):
        words = set()
        xArr = X.values
        for x in xArr:
            counts = Counter(str(x).split())
            for word in counts.keys():
                words.add(word)
        return len(words)

    # @profile
    def fit(self, X, y):

        # dataframe to npArray
        # inside try catch, otherwise it crashes when using bagging
        try:
            xArr = X.values
        except:
            xArr = X
        try:
            yArr = y.values
        except:
            yArr = y

        # total amount of tweets
        n = X.shape[0]

        # storing the amount of tweets that are 'positive' or 'negative'
        self.tweets_positive = np.count_nonzero(yArr)
        self.tweets_negative = n - self.tweets_positive

        # calculating the prior probability of a message being 'positive' or 'negative'
        self.log_prior_probability['negative'] = math.log(self.tweets_negative)
        self.log_prior_probability['positive'] = math.log(self.tweets_positive)

        self.wc['negative'] = {}
        self.wc['positive'] = {}

        # iterating through all tweets
        for x, y in zip(xArr, yArr):

            c = 'negative'
            if y == 1:
                c = 'positive'

            # splitting tweet into dictionary word:count
            xsplit = str(x).split()
            counts = Counter(xsplit)

            # for each unique word in the tweet
            for word, count in counts.items():

                # creating new items should they not exist
                if len(self.dictionary) >= self.dictionary_size:
                    break
                self.dictionary.add(word)
                if word not in self.wc[c]:
                    self.wc[c][word] = 0

                # increasing the count for word in class
                self.wc[c][word] += count

    # @profile
    def predict(self, X):
        # dataframe to npArray
        # inside try catch, otherwise it crashes when using bagging
        try:
            xArr = X.values
        except:
            xArr = X

        result = []
        dictionary_length = len(self.dictionary)
        tweet_num_positive = self.tweets_positive
        tweet_num_negative = self.tweets_negative
        log_prior_probability_positive = self.log_prior_probability['positive']
        log_prior_probability_negative = self.log_prior_probability['negative']

        for tweet in xArr:

            xsplit = str(tweet).split()
            word_counts = Counter(xsplit)
            positive_count = 0
            negative_count = 0
            for word, _ in word_counts.items():
                if word not in self.dictionary:
                    continue

                # Applying Naive Bayes
                # We need to calculate p(w_i | positive) and p(w_i | negative)
                # The numerator is how many times w_i appears in a tweet of such class, divided by the count of
                # all words in the tweets of the class.
                # Since we can't calculate log of 0, we use Laplace Smoothing, adding l to the numerator.
                # In order to balance it, the size of the dictionary must be added to the denominator

                # using get instead of [], we avoid error when the key does not exist and we can also set a default value
                numerator_positive = self.wc['positive'].get(word, 0) + self.laplace_smoothing
                numerator_negative = self.wc['negative'].get(word, 0) + self.laplace_smoothing

                # should there be a 0, we don't take this word into consideration (only used when laplace smoothing == 0 as well)
                if numerator_positive == 0 or numerator_negative == 0:
                    continue

                log_positive = math.log(numerator_positive
                                        / (tweet_num_positive + dictionary_length))
                log_negative = math.log(numerator_negative
                                        / (tweet_num_negative + dictionary_length))

                positive_count += log_positive
                negative_count += log_negative

            # Adding prior probability of each class
            positive_count += log_prior_probability_positive
            negative_count += log_prior_probability_negative

            if positive_count > negative_count:
                result.append(1)
            else:
                result.append(0)

        return result
