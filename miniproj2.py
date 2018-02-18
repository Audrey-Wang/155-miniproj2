import numpy as np
import matplotlib.pyplot as plt

data   = np.genfromtxt("data/data.txt").astype(int) # User Id, Movie Id, Rating
movies = np.genfromtxt("data/movies.txt", delimiter="\t",
                    usecols=[x for x in range(21) if x != 1]).astype(int)
movie_names = np.genfromtxt("data/movies.txt", delimiter="\t",
                    usecols = [1], dtype=str)
params = ["Movie Id",
          "Unknown",
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
test   = np.genfromtxt("data/test.txt")
train  = np.genfromtxt("data/train.txt")

plt.hist(data[:, 2], bins='auto')
plt.title("All Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.xticks([1, 2, 3, 4, 5])
plt.show()