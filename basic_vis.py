# Functions for basic visualization.
import numpy as np
import matplotlib.pyplot as plt

# Saving average scores, # of ratings for each movie.
def summarize(movie_titles, data):
    ratings = open('data/summary.txt', 'w')
    ratings.write("movie_id average_rating\n")
    for i in range(len(movie_titles)):
        all_ratings = [row[2] for row in data if row[1] == (i + 1)]
        rating = np.mean(all_ratings)
        ratings.write("%d %f\n" % (len(all_ratings), rating))
    ratings.close()

def basic_visualization(data, plot_title, fname, movie_ids=None):
    if movie_ids is None:
        plt.hist(data[:, 2], bins='auto')
    else:
        plt.hist([row[2] for row in data if (row[1] - 1) in movie_ids], 
                bins='auto')
    plt.title(plot_title)
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks([1, 2, 3, 4, 5])
    plt.savefig(fname)	
    plt.clf()