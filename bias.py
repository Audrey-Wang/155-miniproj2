import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import PredefinedKFold
from matrix_vis import visualization

u_cols = ['movie_id', 'movie_title', 'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
# users = pd.read_csv('data/movies.txt', sep='\t', names = u_cols, encoding = 'latin-1')

r_cols = ['user_id', 'movie_id', 'rating']
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_folds([('data/train.txt', 'data/test.txt')], reader=reader)
pkf = PredefinedKFold()

for trainset, testset in pkf.split(data):
    algo = SVD()
    algo.fit(trainset)
    u = algo.pu
    v = algo.qi
    v = np.transpose(v)
    print(u.shape)
    print(v.shape)
    a, _, _ = np.linalg.svd(v)
    a = a[:2]
    vplot = np.dot(a, v)
    print(vplot.shape)

    movie_ratings = np.genfromtxt("data/summary.txt",  names=True)
    movie_titles = []
    movie_data = []
    with open("data/movies.txt", mode="r", encoding="ISO-8859-1") as f:
        for line in f:
            line = line.split("\t")
            movie_titles.append(line[1])
            movie_data.append(line[2:])
    movie_data = np.array(movie_data).astype(int)

    chosen = np.array([1, 2, 5, 10, 50, 100, 200, 500, 1000, 1500])
    #visualization(users[:,0], users[:,1], vt, "plot", 'plot.png')
    most_popular = np.argpartition(movie_ratings['num_ratings'], -10)[-10:].tolist()
    best_rated = np.argpartition(
        movie_ratings['average_rating'], -10)[-10:].tolist()
    #best_rated = np.argpartition(ratings['average_rating'], -10)[-10:].tolist()
    genres = [1, 3, 5]
    # (a) Any ten movies
    visualization(np.array(chosen), movie_titles, vplot, "Ten Movies of Choice", "5_3_a.png")
    # (b) Ten most popular movies
    visualization(most_popular, movie_titles, vplot, "Ten Most Popular Movies", "5_3_b.png")
    # (c) Ten best movies 
    visualization(best_rated, movie_titles, vplot, "Ten Best Movies", "5_3_c.png")
    # (d) Ten movies from selected genres.

    for i in genres:
        movie_ids = []
        for j in range(len(movie_data)):
            if movie_data[j][i] == 1:
                # print(users[:,1][j])
                # print(movie_data[j])
                movie_ids.append(j)
        visualization(movie_ids[:10], movie_titles, vplot, 
            "Ten %s Movies" % u_cols[i + 2], "5_3_d_%d.png" % i)