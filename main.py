import os
import re
import string
import math
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

        self.tweet_num = {}
        self.log_class_priors = {}
        self.wc = {}
        self.vocab = set()

    def item_count(self, text):
        wc = {}
        for w in text:
            wc[w] = wc.get(w, 0.0) + 1.0
        return wc

    def split(self, text):
        return re.split("\W+", text)

    def fit(self, X, y):


        xArr = X.values
        yArr = y.values

        n = X.shape[0]
        self.tweet_num['negative'] = sum(1 for c in yArr if c == 0)
        self.tweet_num['positive'] = sum(1 for c in yArr if c == 1)
        self.log_class_priors['negative'] = math.log(self.tweet_num['negative'])
        self.log_class_priors['positive'] = math.log(self.tweet_num['positive'])
        self.wc['negative'] = {}
        self.wc['positive'] = {}

        for x, y in zip(xArr, yArr):

            c = 'negative'
            if y == 1:
                c = 'positive'

            counts = self.item_count(self.split(x))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if  word not in self.wc[c]:
                    self.wc[c][word] = 0.0


        return n

def main():
    X, y = get_data()
    nb = naiveBayes()
    nb.fit(X, y)
    print("hola")
if __name__ == "__main__":
    main()