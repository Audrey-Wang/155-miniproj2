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

def basic_visualizations(movie_ratings, movie_data, data, params):
    # 1. all ratings in the movielens dataset.
    plt.hist(data[:, 2], bins='auto')
    plt.title("All Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks([1, 2, 3, 4, 5])
    plt.savefig("4_1.png")	
    plt.clf()

    # 2. all ratings of the ten most popular movies.
    most_popular = np.argpartition(movie_ratings['num_ratings'], -10)[-10:].tolist()
    plt.hist([row[2] for row in data if (row[1] - 1) in most_popular], bins='auto')
    plt.title("All Ratings for Ten Most Popular Movies")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks([1, 2, 3, 4, 5])
    plt.savefig("4_2.png")
    plt.clf()

    # 3. all ratings of the ten best movies 
    best_rated = np.argpartition(
        movie_ratings['average_rating'], -10)[-10:].tolist()
    plt.hist([row[2] for row in data if (row[1] - 1) in best_rated], bins='auto')
    plt.title("All Ratings for Ten Best Movies")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks([1, 2, 3, 4, 5])
    plt.savefig("4_3.png")
    plt.clf()

    # 4. all ratings of movies from three genres of your choice.
    genres = [1, 3, 5]
    for i in genres:
        to_plot = []
        for j in range(len(movie_data)):
            if movie_data[j][i] == 1:
                to_plot += [row[2] for row in data if (row[1] - 1) == j]
        plt.hist(to_plot)
        plt.title("All Ratings for %s Movies" % params[i])
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.xticks([1, 2, 3, 4, 5])
        plt.savefig("4_4_%d.png" % i)
        plt.clf()