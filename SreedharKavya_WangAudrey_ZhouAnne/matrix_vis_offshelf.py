# Functions for matrix factorization visualizations.
import numpy as np
import matplotlib.pyplot as plt

#Revision History: 
#   Anne created
#   Kavya added -1 indices for off-shelf implementation
def visualization(movie_ids, movie_titles, V, plot_title, fname):
    for i in movie_ids:
        plt.plot(V[0][i-1], V[1][i-1], marker='o') 
        plt.text(V[0][i-1], V[1][i-1], movie_titles[i-1])
    plt.title(plot_title)
    plt.savefig(fname)  
    plt.clf()