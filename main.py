import re
import math
import sys
import time

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pandas
import sklearn
import argparse
from collections import Counter
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns



def read_data(n_rows=None):
    # data = pandas.read_csv('top10k.csv', sep=';')
    data = pandas.read_csv('data.csv', sep=';')
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
    def __init__(self, laplace_smoothing=1, dictionary_size=sys.maxsize):

        self.laplace_smoothing = laplace_smoothing
        self.tweet_num = {}  # stores the amount of tweets that are 'positive' or 'negative'
        self.log_prior_probability = {}  # stores the log prior probability of a message being 'positive' or 'negative'
        self.wc = {}  # for each class, stores the amount of times a word appears
        self.dictionary = set()  # set of unique words.
    #     TODO canviar set per list i fer el :dictionary_size
        self.dictionary_size = dictionary_size


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
            xsplit = str(x).split()
            counts = Counter(xsplit)

            # for each unique word in the tweet
            for word, count in counts.items():

                # creating new items should they not exist
                # if word not in self.dictionary:
                if len(self.dictionary) >= self.dictionary_size:
                    break
                self.dictionary.add(word)
                if word not in self.wc[c]:
                    self.wc[c][word] = 0.0

                # increasing the count for word in class
                self.wc[c][word] += count

    # @profile
    def predict(self, X):
        xArr = X.values
        result = []

        for tweet in xArr:

            xsplit = str(tweet).split()
            word_counts = Counter(xsplit)
            positive_count = 0
            negative_count = 0
            for word, _ in word_counts.items():
                if word not in self.dictionary:
                    continue

                # TODO canviar el .get per un []
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


# @profile
def main():
    args = parse_arguments()

    startTime = time.time()


    X, y = read_data(n_rows=args.n_rows)
    # X, y = read_data(n_rows=100000)

    test(X, args, y)

    # test_size_comparison(X, args, y)
    # dictionary_length_comparison(X, args, y, partitions=20)

    # kfold(X, args, y)

    print("Elapsed time: ", time.time() - startTime)


def test_size_comparison(X, args, y):
    test_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.98]
    results = []
    for i in test_sizes:
        results.append(test(X, args, y, test_size=i))
    show_bar_plot(results, test_sizes, 'Accuracy', 'Test Size Comparison')

def dictionary_length_comparison(X, args, y, partitions=10):
    sizes = []
    sizesLabel = []
    maxSize = naiveBayes.count_total_words(X)

    mult = 100
    for i in range(partitions):
        sizes.append(maxSize)
        sizesLabel.append(str("%.2f" % mult))
        maxSize = int(maxSize/2)
        mult = mult/2

    # for i in range(1, partitions+1):
    #     sizes.append(int(i*maxSize / partitions))
    #     sizesLabel.append(str(i*10)+'%')

    results = []
    for s in sizes:
        results.append(test(X, args, y, dictionary_size=s))
    show_bar_plot(results, sizesLabel, 'Accuracy', 'dictionary Size Comparison', xlabel='Size percentatge')


def test(X, args, y, test_size=0.2, dictionary_size=sys.maxsize):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    nb = naiveBayes(laplace_smoothing=args.smooth, dictionary_size=dictionary_size)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    # conf_matrix(y_test, y_pred)
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)


def kfold(X, args, y):
    kf = KFold(n_splits=args.n_splits, random_state=None, shuffle=True)
    NB = naiveBayes(laplace_smoothing=args.smooth)
    # metrics = ('accuracy', 'precision', 'recall', 'f1_micro')
    metrics = ('accuracy')
    cv_results = cross_validate(NB, X, y, cv=kf, scoring=metrics)
    # for metric in list(metrics):
    print('accuracy average: ', cv_results['test_'+str('score')].mean())
    print('accuracy standard deviation: ', np.std(cv_results['test_'+str('score')]))


def conf_matrix(true, pred):
    cf_matrix = confusion_matrix(true, pred, normalize='true')
    ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()

def show_bar_plot(values, labels, ylabel='', title='', xlabel=''):
    xlabels = labels
    yvalues = values
    x_pos = [i for i, _ in enumerate(xlabels)]
    plt.bar(x_pos, yvalues)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xticks(x_pos, xlabels)
    plt.ylim([0.5, 0.8])
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', type=float, default=1, help='Value for Laplace Smoothing')
    parser.add_argument('--n_rows', type=int, default=None, help='Amount of rows to read from csv file')
    parser.add_argument('--n_splits', type=int, default=5, help='K_Fold splits')
    # TODO add arguments test size i dictionary size
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
