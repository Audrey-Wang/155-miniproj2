# Functions for matrix factorization visualizations.
import numpy as np
import matplotlib.pyplot as plt

def visualization(movie_ids, movie_titles, V, plot_title, fname):
    for i in movie_ids:
        plt.plot(V[0][i], V[1][i], marker='o') 
        plt.text(V[0][i], V[1][i], movie_titles[i])
    plt.title(plot_title)
    plt.savefig(fname)	
    plt.clf()