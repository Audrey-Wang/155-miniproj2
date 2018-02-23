import numpy as np
from basic_vis import summarize, basic_visualization
from matrix_vis import visualization
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
# summarize(movie_titles, data)
movie_ratings = np.genfromtxt("data/summary.txt",  names=True)
params = ["Unknown",
          "Action",
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

# Indices of specific movies for visualizations.
chosen = [1, 2, 5, 10, 50, 100, 200, 500, 1000, 1500]
most_popular = np.argpartition(movie_ratings['num_ratings'], -10)[-10:].tolist()
best_rated = np.argpartition(
    movie_ratings['average_rating'], -10)[-10:].tolist()
genres = [1, 3, 5]

# Parameters for matrix factorization.
m = np.amax(data, axis=0)[0]    # number of users 
n = len(movie_titles)           # number of movies
k = 20
eta = 0.03
reg = 0.01

# BASIC VISUALIZATIONS
# 1. all ratings in the movielens dataset.
# basic_visualization(data, "All Movie Ratings", "4_1.png")
# # 2. all ratings of the ten most popular movies.
# basic_visualization(data, "All Ratings for Ten Most Popular Movies", 
#   "4_2.png", movie_ids=most_popular)
# # 3. all ratings of the ten best movies 
# basic_visualization(data, "All Ratings for Ten Best Movies", "4_3.png", 
#   movie_ids=best_rated)
# # 4. all ratings of movies from three genres of your choice.
# for i in genres:
#     movie_ids = []
#     for j in range(len(movie_data)):
#         if movie_data[j][i] == 1:
#             movie_ids.append(j)
#     basic_visualization(data, "All Ratings for %s Movies" % params[i],  
#         "4_4_%d.png" % i, movie_ids=movie_ids)

# MATRIX FACTORIZATION VISUALIZATIONS
# Generate V-tilde using all 3 methods. 
# Method 1.
U1, V1, _ = train_model(m, n, k, eta, reg, train)
test_err1 = get_err(U1, V1, test, reg)
a, _, _ = np.linalg.svd(V1)
a = a[:2]
V1_plot = np.dot(a, V1)

# Method 2. 

# Method 3.

# Do visualizations (using V1_plot, V2_plot, and V3_plot)
# (a) Any ten movies
visualization(chosen, movie_titles, V1_plot, "Ten Movies of Choice", "5_1_a.png")
# (b) Ten most popular movies
visualization(most_popular, movie_titles, V1_plot, "Ten Most Popular Movies", "5_1_b.png")
# (c) Ten best movies 
visualization(best_rated, movie_titles, V1_plot, "Ten Best Movies", "5_1_c.png")
# (d) Ten movies from selected genres.
for i in genres:
    movie_ids = []
    for j in range(len(movie_data)):
        if movie_data[j][i] == 1:
            movie_ids.append(j)
    visualization(movie_ids[:10], movie_titles, V1_plot, 
        "Ten %s Movies" % params[i], "5_1_d_%d.png" % i)