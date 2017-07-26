
# coding: utf-8

# # MovieLens recommender system using SVD matrix decomposition

import pandas as pd
import numpy as np
#import tmdbsimple as tmdb
from sklearn.pipeline import make_pipeline
from sklearn import pipeline, feature_selection, decomposition
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import pprint
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds


# In[3]:


ratings=pd.read_csv('../../data/ratings.csv')
movies=pd.read_csv('../../data/movies.csv')
tags=pd.read_csv('../../data/tags.csv')
genome_scores=pd.read_csv('../../data/genome-scores.csv')
genome_tags=pd.read_csv('../../data/genome-tags.csv')


# In[4]:

ratings.head()


# In[5]:

movies.head()


# Create dictionary of Movie ID's and Titles so that we can have titles in Ratings dataframe:

# In[6]:

dictionary=movies.set_index('movieId').to_dict()['title']


# In[7]:

dictionary


# In[8]:

ratings['movieName']=ratings['movieId'].map(dictionary)


# In[9]:

ratings.head()


# Save this new DataFrame to a csv file for future use:

# In[10]:

#ratings.to_csv('newDF.csv', sep=',', index=False)


# Use only a subset of movies for computation reasons - using top 600 movies:

# In[11]:

n=600
top_n = ratings.movieId.value_counts().index[:n]
ratings = ratings[ratings.movieId.isin(top_n)]
ratings.head()


# Create wide matrix:

# In[12]:

wideMatrix = pd.pivot_table(ratings,values='rating',
                                index=['userId','movieId'],
                                aggfunc=np.mean).unstack()


# In[13]:

wideMatrix.ix[0:5, 0:5]


# ** Fill NaN values with 0: **

# In[14]:

wideMatrix2=wideMatrix.fillna(0)


# In[15]:

wideMatrix2.head()


# ** de-mean  **

# In[16]:

R = wideMatrix2.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


# ** SVD decomposition of the demeaned matrix **

# In[17]:

U, sigma, Vt = svds(R_demeaned, k = 50)


# In[18]:

sigma = np.diag(sigma)


# ** dot product $U \Sigma V^T$ to get approximation matrix **

# In[19]:

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = wideMatrix2.columns)


# In[20]:

ratings.head()


# In[20]:

def recommend_movies(predictions_df, userId, movies_df, original_ratings_df, num_recommendations=5):

    # Get and sort the user's predictions
    user_row_number = userId- 1 # userId starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userId)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userId, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# list of user's we can check with SVD and then compare with Factorization Machine method:
#
# check=[8405, 34576, 59477, 74142, 79159, 82418, 118205, 121535, 125794, 131904]

# In[21]:

already_rated, predictions = recommend_movies(preds_df, 125794, movies, ratings, 10)


# In[22]:

already_rated.head(10)


# In[23]:

print predictions

