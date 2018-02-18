import numpy as np
import matplotlib.pyplot as plt

# Import all the data.
# for all movie_ arrays, (index + 1) = id of the movie it corresponds to.
# @ movie_titles: string array of movie titles. 
# @ movie_data: numpy matrix of movie data (what params they satisfy) 
# @ movie_ratings: numpy array of # of ratings + average rating per movie.
# @ movie_counts: int numpy array of # of ratings for that movie.
data = np.genfromtxt("data/data.txt").astype(int) # User Id, Movie Id, Rating
movie_titles = []
movie_data = []
with open("data/movies.txt", mode="r", encoding="ISO-8859-1") as f:
    for line in f:
        line = line.split("\t")
        movie_titles.append(line[1])
        movie_data.append(line[2:])
movie_data = np.array(movie_data).astype(int)
movie_ratings = np.genfromtxt("data/summary.txt",  names=True)
params = ["Action",
          "Adventure",
          "Animation",
          "Childrens",
          "Comedy",
          "Crime",
          "Documentary",
          "Drama",
          "Fantasy",
          "Film-Noir",
          "Horror",
          "Musical",
          "Mystery",
          "Romance",
          "Sci-Fi",
          "Thriller",
          "War",
          "Western"]
test   = np.genfromtxt("data/test.txt")
train  = np.genfromtxt("data/train.txt")

# BASIC VISUALIZATIONS
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