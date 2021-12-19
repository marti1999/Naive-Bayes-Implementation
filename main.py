import time

import pandas
import argparse
from tests import *


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


# @profile
def main():
    args = parse_arguments()

    startTime = time.time()


    # X, y = read_data(n_rows=args.n_rows)
    X, y = read_data(n_rows=100000)

    # test(X, args, y)

    laplace_smoothing_comparison(X, args, y)
    # test_size_comparison(X, args, y)
    # dictionary_length_comparison(X, args, y, partitions=10)

    # kfold(X, args, y)

    print("Elapsed time: ", time.time() - startTime)


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
