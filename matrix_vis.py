# Functions for matrix factorization visualizations.
import numpy as np
import matplotlib.pyplot as plt

# Saving average scores, # of ratings for each movie.
def summarize(movie_titles, data, fname):
    ratings = open(fname, 'w')
    ratings.write("movie_id average_rating\n")
    for i in range(len(movie_titles)):
        all_ratings = [row[2] for row in data if row[1] == (i + 1)]
        rating = np.mean(all_ratings)
        ratings.write("%d %f\n" % (len(all_ratings), rating))
    ratings.close()