import numpy as np
from basic import summarize, basic_visualizations

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
# summarize(movie_titles, data)
# basic_visualizations(movie_ratings, data)