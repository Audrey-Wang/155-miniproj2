import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('data/data.txt', sep='\t', names=r_cols, encoding='latin-1')
ratings = np.array(ratings)
users = ratings[ratings[:,1].argsort()]
print(ratings)

print(users)