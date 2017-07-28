import pandas as pd
import numpy as np

# building rating matrix
df = pd.read_csv('ratings.csv', sep=',')
df_id = pd.read_csv('links.csv', sep=',')
df = pd.merge(df, df_id, on=['movieId'])

rating_matrix = np.zeros((df.userId.unique().shape[0], max(df.movieId)))
for row in df.itertuples():
    rating_matrix[row[1]-1, row[2]-1] = row[3]
rating_matrix = rating_matrix[:,:9000]