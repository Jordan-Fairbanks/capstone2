import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from ..utils import labels_to_numbers

def plot_2d(ax,X,y=None, title='', colors=None, categories=None, perplexity=45):
    """
    plot_2d takes an ndarray, a title (and, optionally, a set of target labels, a dictionary 
    of colors, and a dictionary of categories), and uses t-SNE to reduce the dimesionality of 
    the matrix, then plots and returns a matplotlib object.
    INPUTS:
        X - numpy.ndarray - input matrix to be reduced and represented
        y - list, series, or array like iterable of integers - target labels for each row of X
        title - str - a string to represent the title of the graph
        colors - dict - a dictionary of colors corresponding to each target label 
        categories - dict - a dictionary of category names corresponding to each label
    RETURNS:
    <class 'matplotlib.figure.Figure'> - 2d plot of t-SNE reduced matrix X
    """

    tsne = TSNE(n_components=2, perplexity=perplexity)
    two_dimensional = tsne.fit_transform(X)
    if y and colors:
        for first, second, group in zip(two_dimensional[:,0], two_dimensional[:,1], y):
            ax.scatter(first, second, color=colors[group], alpha=.3)
    else:
        for first, second in zip(two_dimensional[:,0], two_dimensional[:,1]):
            ax.scatter(first, second, alpha=.3)
    ax.set_title(title, fontsize=16, weight='bold')
    if colors and categories:  
        handles = [mpatches.Patch(color=colors[key], label=val) for key, val in categories.items()]
        ax.legend(handles=handles)
    return ax


def plot_3d(ax,X,y=None, title='', colors=None, categories=None, perplexity=50, window=None):
    """
    plot_3d takes an ndarray, a title (and, optionally, a set of target labels, a dictionary 
    of colors, and a dictionary of categories), and uses t-SNE to reduce the dimesionality of 
    the matrix, then plots and returns a matplotlib object.
    INPUTS:
        ax - matplotlib axes obj - the axes object on which the graph is plotted
        X - numpy.ndarray - input matrix to be reduced and represented
        y - list, series, or array like iterable of integers - target labels for each row of X
        title - str - a string to represent the title of the graph
        colors - dict - a dictionary of colors corresponding to each target label 
        categories - dict - a dictionary of category names corresponding to each label
        window - dict - a dictionary of lower and upper limits for each dimension
                        ex. {'x':(0,10),
                             'y':(-1,1),
                             'z':(100,200)}
    RETURNS:
        <class 'matplotlib.figure.Figure'> - 3d plot of t-SNE reduced matrix X
    """

    tsne = TSNE(n_components=3, perplexity=perplexity)
    three_dimensional = tsne.fit_transform(X)

    if y and colors:
        for first,second,third, group in zip(three_dimensional[:,0],three_dimensional[:,1],three_dimensional[:,2],y):
            ax.scatter(first, second, third, color=colors[group], alpha=.3)

    else:
        for first,second,third in zip(three_dimensional[:,0],three_dimensional[:,1],three_dimensional[:,2]):
            ax.scatter(first, second, third, alpha=.3)
    ax.set_title(title, fontsize=16, weight='bold')

    if window:
        ax.set_xlim(window['x'][0], window['x'][1])
        ax.set_ylim(window['y'][0], window['y'][1])
        ax.set_zlim(window['z'][0], window['z'][1])

    if colors and categories:
        handles = [mpatches.Patch(color=colors[key], label=val) for key, val in categories.items()]
        ax.legend(handles=handles, loc=4)
    return ax

if __name__ == '__main__':
    df = pd.read_parquet('train')
    y = df.pop('Target').values
    X = df.values

    colors = {0:'red',
              1:'blue',
              2:'green',
              3:'orange',
              4:'purple',}
    categories, y = labels_to_numbers(y)


    fig, ax = plt.subplots(figsize=(10,10))
    ax = plot_2d(ax, X, y,'t-SNE Mapping of Vectorized Documents (2d)', colors, categories)
    plt.save_fig('tsne-2d.png')

    window = {'x':(-250,-50),
              'y':(200,450),
              'z':(-250,-50)}

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax = plot_3d(ax, X, y, 't-SNE Mapping of Vectorized Documents (3d)', colors, categories, window=window)
    plt.save_fig('tsne-3d.png')