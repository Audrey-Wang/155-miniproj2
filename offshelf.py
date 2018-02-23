import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from matrix_vis_offshelf import *

u_cols = ['movie_id', 'movie_title', 'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
users = pd.read_csv('data/movies.txt', sep='\t', names = u_cols, encoding = 'latin-1')

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('data/data.txt', sep='\t', names=r_cols, encoding='latin-1')
most = np.copy(ratings)
'''
print(users.shape)
#print(users.head())
print(ratings.shape)
#print(ratings.head())
'''

test   = np.genfromtxt("data/test.txt").astype(int)
train  = np.genfromtxt("data/train.txt").astype(int)

#test = np.reshape(test, (10000, -1)) 
train = np.reshape(train, (90000, -1))
train_data_matrix = np.zeros(shape=(943,1682))
train_data = pd.DataFrame(train)
for line in train_data.itertuples():
	train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

movie_ratings = np.genfromtxt("data/summary.txt",  names=True)
users = np.array(users)
visualization(users[:,0], users[:,1], vt, "plot", 'plot.png')

chosen = np.array([1, 2, 5, 10, 50, 100, 200, 500, 1000, 1500])
#visualization(users[:,0], users[:,1], vt, "plot", 'plot.png')
most_popular = np.argpartition(movie_ratings['num_ratings'], -10)[-10:].tolist()
best_rated = np.argpartition(
    movie_ratings['average_rating'], -10)[-10:].tolist()
#best_rated = np.argpartition(ratings['average_rating'], -10)[-10:].tolist()
genres = [1, 3, 5]
# (a) Any ten movies
visualization(np.array(chosen), users[:,1], vt, "Ten Movies of Choice", "5_2_a.png")
# (b) Ten most popular movies
visualization(most_popular, users[:,1], vt, "Ten Most Popular Movies", "5_2_b.png")
# (c) Ten best movies 
visualization(best_rated, users[:,1], vt, "Ten Best Movies", "5_2_c.png")
# (d) Ten movies from selected genres.

movie_data = []
with open("data/movies.txt", mode="r", encoding="ISO-8859-1") as f:
    for line in f:
        line = line.split("\t")
        movie_data.append(line[2:])
movie_data = np.array(movie_data).astype(int)

for i in genres:
    movie_ids = []
    for j in range(len(movie_data)):
        if movie_data[j][i] == 1:
        	movie_ids.append(j)
    visualization(movie_ids[:10], users[:,1], vt, 
        "Ten %s Movies" % u_cols[i], "5_2_d_%d.png" % i)