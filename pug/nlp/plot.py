
import seaborn as sb
import pandas
np = pandas.np
import bisect

from matplotlib import pyplot as plt


# from pylab import figure, savefig, imshow, axes, axis, cm, show




#####################################################################################
######## Based on the statistics plotting wrapper from Udacity ST-101
######## https://www.udacity.com/wiki/plotting_graphs_with_python


def scatterplot(x, y):
    plt.ion()
    plt.plot(x, y, 'b.')
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.draw()


def barplot(labels, data):
    pos = np.arange(len(data))
    plt.ion()
    plt.xticks(pos + 0.4, labels)
    plt.bar(pos, data)
    plt.grid('on')
    #plt.draw()


def histplot(data, bins=None, nbins=5):