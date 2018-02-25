import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import PredefinedKFold
from matrix_vis import visualization

data = np.genfromtxt("data/data.txt").astype(int) # User Id, Movie Id, Rating
movie_titles = []
movie_data = []
with open("data/movies.txt", mode="r", encoding="ISO-8859-1") as f:
    for line in f:
        line = line.split("\t")
        movie_titles.append(line[1])
        movie_data.append(line[2:])
movie_data = np.array(movie_data).astype(int)
user_ratings = np.genfromtxt("data/user_summary.txt",  names=True)

p = Polynomial.fit(user_ratings['average_rating'], user_ratings['num_ratings'], 4)
fig, ax = plt.subplots()
ax.stem(user_ratings['average_rating'], user_ratings['num_ratings'])
plt.title("Number of Ratings vs. Average Rating for Users")
plt.xlabel("Average Rating")
plt.ylabel("Number of Ratings")
plt.xticks([1, 2, 3, 4, 5])
# ax2 = ax.twinx()
# ax2.plot(*p.linspace(), 'g-')
# ax2.get_yaxis().set_visible(False)
plt.savefig("submission.png")	
plt.clf()
