import os
import re
import string
import math
import pandas


def get_data():
    data = pandas.read_csv('top10k.csv', sep=';')
    X = data['tweetText']
    y = data['sentimentLabel']
    # return X.values, y.values
    return X.tolist(), y.tolist()



def main():
    X, y = get_data()
    print("hola")
if __name__ == "__main__":
    main()