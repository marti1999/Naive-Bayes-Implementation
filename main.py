import re
import math
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import classification_report
import numpy as np
import pandas
import sklearn
import argparse


def get_data(n_rows=None):
    # data = pandas.read_csv('top10k.csv', sep=';')
    data = pandas.read_csv('FinalStemmedSentimentAnalysisDataset.csv', sep=';')
    data = data.sample(random_state=0, frac=1).reset_index(drop=True)

    if n_rows is not None and n_rows > 0:
        data = data.sample(n=min(n_rows, data.shape[0])).reset_index(drop=True)

    X = data['tweetText']
    y = data['sentimentLabel']
    # return X.values, y.values
    # return X.tolist(), y.tolist()
    return X, y


# Inherit from sklearn.base.BaseEstimator
# https://scikit-learn.org/stable/developers/develop.html
# Needed in order to use some sklearn modules like cross_validate
#   "All estimators in the main scikit-learn codebase should inherit from sklearn.base.BaseEstimator."
class naiveBayes(sklearn.base.BaseEstimator):
    def __init__(self, laplace_smoothing=1):
        if laplace_smoothing is None:
            laplace_smoothing = 1
        self.laplace_smoothing = laplace_smoothing
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
        return re.split("\W+", str(text))

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
            # TODO: intentar utilitzar la llibreria Counter
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
        xArr = X.values
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
                log_positive = math.log((self.wc['positive'].get(word, 0.0) + self.laplace_smoothing)
                                        / (self.tweet_num['positive'] + len(self.dictionary)))
                log_negative = math.log((self.wc['negative'].get(word, 0.0) + self.laplace_smoothing)
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
    args = parse_arguments()

    X, y = get_data(n_rows=args.n_rows)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # nb = naiveBayes(laplace_smoothing=args.smooth)
    # nb.fit(X_train, y_train)
    # y_pred = nb.predict(X_test)
    # print(classification_report(y_test, y_pred))

    kf = KFold(n_splits=args.n_splits, random_state=None, shuffle=True)
    NB = naiveBayes(laplace_smoothing=args.smooth)
    metrics = ('accuracy', 'precision', 'recall', 'f1_micro')
    cv_results = cross_validate(NB, X, y, cv=kf, scoring=metrics)
    for metric in list(metrics):
        print(metric, cv_results['test_'+str(metric)].mean())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', type=float, default=1, help='Value for Laplace Smoothing')
    parser.add_argument('--n_rows', type=int, default=None, help='Amount of rows to read from csv file')
    parser.add_argument('--n_splits', type=int, default=5, help='K_Fold splits')
    args = parser.parse_args()
    if args.smooth is not None and args.smooth < 0:
        parser.error("smooth argument cannot be less than 0")
    if args.n_rows is not None and args.n_rows < 1:
        parser.error("n_rows cannot be less than 1")
    if args.n_splits is not None and (args.n_splits < 1 or args.n_splits > 50):
        parser.error("n_splits must be between 1 and 50")
    return args


if __name__ == "__main__":
    main()
