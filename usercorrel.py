import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('data/data.txt', sep='\t', names=r_cols, encoding='latin-1')
ratings = np.array(ratings)
users = ratings[ratings[:,1].argsort()]

cnts = [0]*1682
for i in range(0, users.shape[0]):
	cnts[users[i][1] - 1] += 1

summ = pd.read_csv('data/summary.txt', sep='|', encoding='latin-1')
print(summ)
print("SSSSSSSSSSS")
summ = np.array(summ).tolist()

print(len(summ))
# for i in range(0, len)

# plt.plot(summ[0, :], summ[1, :])
# plt.show()