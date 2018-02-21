import numpy as np
from basic_vis import summarize, basic_visualizations
from matrix_vis import summarize
from hw5utils import train_model, get_err

# Import all the data.
# for all movie_ arrays, (index + 1) = id of the movie it corresponds to.
# @ movie_titles: string array of movie titles. 
# @ movie_data: numpy matrix of movie data (what params they satisfy) 
# @ movie_ratings: numpy array of # of ratings + average rating per movie.
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
test   = np.genfromtxt("data/test.txt").astype(int)
train  = np.genfromtxt("data/train.txt").astype(int)
# for some reason test and train aren't multi-d arrays, but data is?
test = np.reshape(test, (10000, -1)) 
train = np.reshape(train, (90000, -1))

# Parameters for matrix factorization.
m = np.amax(data, axis=0)[0]    # number of users 
n = len(movie_titles)           # number of movies
k = 20
eta = 0.03
reg = 0.01

# BASIC VISUALIZATIONS
# summarize(movie_titles, data)
# basic_visualizations(movie_ratings, movie_data, data, params)

# MATRIX FACTORIZATION VISUALIZATIONS
# Model 1.
U1, V1, _ = train_model(m, n, k, eta, reg, train)
test_err1 = get_err(U1, V1, test, reg)
print(test_err1)
a, _, _ = np.linalg.svd(V1)
a = a[:2]
V1_plot = np.dot(a, V1)
print(V1_plot)