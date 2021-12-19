import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def conf_matrix(true, pred):
    cf_matrix = confusion_matrix(true, pred, normalize='true')
    ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    # Display the visualization of the Confusion Matrix.
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
