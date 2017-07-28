# coding: utf-8

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

# used for local
# ratings = pd.read_csv('data/user_log.csv')
# jobs = pd.read_csv('data/job_description_20170331.csv')
# tags = pd.read_csv('data/tags.csv')
# genome_scores = pd.read_csv('data/genome-scores.csv')
# genome_tags = pd.read_csv('data/genome-tags.csv')

#used for amazon
ratings = pd.read_csv('../104/score_log.csv')
jobs = pd.read_csv('../104/job/job_description.csv')

# print all table of rating and jibs
# print ratings.head()
# print jobs.head()


# Create dictionary of Movie ID's and Titles so that we can have titles in Ratings dataframe:
dictionary = jobs.set_index('jobno').to_dict()['job']
print dictionary

ratings['jobName'] = ratings['jobNo'].map(dictionary)
ratings.head()


# Save this new DataFrame to a csv file for future use:

ratings.to_csv('newDF.csv', sep=',', index=False)


# Use only a subset of movies for computation reasons - using top 600 movies:

# In[11]:

n = 600
top_n = ratings.jobNo.value_counts().index[:n]
ratings = ratings[ratings.jobNo.isin(top_n)]
ratings.head()


# Create wide matrix:

# In[12]:

wideMatrix = pd.pivot_table(ratings,values='rating',
                                index=['uid','jobNo'],
                                aggfunc=np.mean).unstack()


# In[13]:

wideMatrix.ix[0:5, 0:5]


# ** Fill NaN values with 0: **

# In[14]:

wideMatrix2=wideMatrix.fillna(0)


# In[15]:

wideMatrix2.head()


# ** de-mean  **


R = wideMatrix2.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


# ** SVD decomposition of the demeaned matrix **

# In[17]:

U, sigma, Vt = svds(R_demeaned, k = 50)


sigma = np.diag(sigma)


# ** dot product $U \Sigma V^T$ to get approximation matrix **


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = wideMatrix2.columns)


ratings.head()



def recommend_movies(predictions_df, uid, movies_df, original_ratings_df, num_recommendations=5):

    # Get and sort the user's predictions
    user_row_number = uid- 1 # userId starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (uid)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'jobNo', right_on = 'jobNo').
                     sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(uid, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['jobNo'].isin(user_full['jobNo'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'jobNo',
               right_on = 'jobNo').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


# list of user's we can check with SVD and then compare with Factorization Machine method:
#
# check=[8405, 34576, 59477, 74142, 79159, 82418, 118205, 121535, 125794, 131904]

# In[21]:

already_rated, predictions = recommend_movies(preds_df, 125794, jobs, ratings, 10)


# In[22]:

already_rated.head(10)


# In[23]:

print predictions