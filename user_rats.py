import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('data/data.txt', sep='\t', names=r_cols, encoding='latin-1')
ratings = np.array(ratings)
users = ratings[ratings[:,0].argsort()]
rats = [0]*943
cnts = [0]*943
for i in range(0, ratings.shape[0]):
	rats[users[i, 0] - 1] += users[i, 2]
	cnts[users[i, 0] - 1] += 1

rats = np.array(rats)
cnts = np.array(cnts)

avgs = rats / cnts

plt.bar(avgs, cnts)
plt.show()