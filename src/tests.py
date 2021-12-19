import sys

import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_validate

from NaiveBayes import naiveBayes
from plotting import show_bar_plot


def test_size_comparison(X, args, y):
    test_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.98]
    results = []
    for i in test_sizes:
        print("test size: ", i, end=" --> accuracy = ")
        results.append(test(X, args, y, test_size=i))
    show_bar_plot(results, test_sizes, 'Accuracy', 'Test Size Comparison')


def laplace_smoothing_comparison(X, args, y):
    values = []
    valuesLabels = []
    lap_smooth= 0.01
    for i in range(10):
        values.append(lap_smooth)
        valuesLabels.append(np.format_float_scientific(lap_smooth, precision=0, exp_digits=1))
        lap_smooth*=10
    results = []
    for v in values:
        args.smooth = v
        print("Laplace Smoothing: ", v, end=" --> accuracy = ")
        results.append(test(X, args, y))
    show_bar_plot(results, valuesLabels, 'Accuracy', 'Laplace Smoothing Comparison', 'alpha')

def test_bagging(X, args, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = BaggingClassifier(base_estimator=naiveBayes(), n_estimators=9, n_jobs=-1, random_state=0).fit(X_train.values.reshape(-1,1), y_train)
    y_pred = clf.predict(X_test.values.reshape(-1,1))
    print(accuracy_score(y_test, y_pred))

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

    results = []
    for s, t in zip(sizes, sizesLabel):
        print("dictionary Length %: ", t, end=" --> accuracy = ")
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