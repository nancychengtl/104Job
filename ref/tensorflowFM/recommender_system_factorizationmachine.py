
# coding: utf-8

# In[1]:

import pandas as pd
from collections import Counter
import tensorflow as tf
from tffm import TFFMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pprint


# Import csv files into dataframes:

# In[2]:

ratings=pd.read_csv('~/ml-20m/ratings.csv', names=['userid', 'movieid','rating', 'timestamp'], skiprows=1)
genres2=pd.read_csv('~/ml-20m/movies.csv', names=['movieid', 'movienm', 'genreid'], skiprows=1)


# Create dictionary of movie id's with their genres so it can be mapped to ratings dataframe and used as context

# In[3]:

dictionary=dict(zip(genres2.movieid,genres2.genreid))


# In[4]:

dictionary


# In[5]:

ratings['genres']=ratings['movieid'].map(dictionary)


# In[6]:

ratings['genres'] = ratings.genres.map(lambda x: x.split('|'))


# Unnest the genres column so there is a row for each movie-genre pair

# In[7]:

def unnest(df, col, reset_index=True):
    col_flat = pd.DataFrame([[i, x] 
                       for i, y in df[col].apply(list).iteritems() 
                           for x in y], columns=['I', col])
    col_flat = col_flat.set_index('I')
    df = df.drop(col, 1)
    df = df.merge(col_flat, left_index=True, right_index=True)
    if reset_index:
        df = df.reset_index(drop=True)
    return df

ratings=unnest(ratings, 'genres')

ratings.head()


# In[8]:

#ratings.to_csv('tffm_df.csv')


# Drop timestamp column:

# In[9]:

ratings.set_index('userid', inplace=True)
ratings=ratings.drop('timestamp', 1)


# ** Using ONLY 10 most common users (for now) because it is so computationally expensive **

# In[10]:

x=Counter(ratings.index).most_common(10)
top_k=dict(x).keys()
ratings=ratings[ratings.index.isin(top_k)]


# In[11]:

ratings.shape


# In[12]:

ratings.head()


# create '_userid' column:

# In[13]:

ratings['_userid']=ratings.index


# In[14]:

ratings.head()


# Convert movieid and genre to 'category' so they can be one-hot encoded:

# In[15]:

ratings['movieid']=ratings['movieid'].astype('category')
ratings['genres']=ratings['genres'].astype('category')
ratings['_userid']=ratings['_userid'].astype('category')


# ** Use Pandas' get_dummies to one-hot-encode the genres and movie ID for each user: **
# 
# 
# 
# **takes about 1min to do the next cell (when using 30 most common users) **

# In[16]:

trans_ratings=pd.get_dummies(ratings)


# In[17]:

trans_ratings['userid2']=trans_ratings.index


# In[18]:

trans_ratings.shape


# In[19]:

trans_ratings.head()


# Set trans_ratings to 'df' for simplicity purposes

# In[20]:

#trans_ratings.to_csv('trans_rating_10.csv')


# In[21]:

df=trans_ratings


# In[22]:

#df.drop(['_userid'], axis=1, inplace=True)


# In[ ]:




# Set X to everything in dataframe (except rating), set y to 'rating'

# In[23]:

X=df.drop(['rating'], axis=1, inplace=False)
y=np.array(df['rating'].as_matrix())


# In[24]:

#X = np.array(X)


# In[25]:

#X = np.nan_to_num(X)


# ** test, train, split **

# In[26]:

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)


# ** X_tr and X_te both contain userId, need to create new df's that don't include this so I can be run into model **

# In[27]:

X_tr.head()


# In[28]:

X_train_withoutUsers=X_tr.drop(['userid2'], axis=1, inplace=False)
X_test_withoutUsers=X_te.drop(['userid2'], axis=1, inplace=False)


# ** create np.array from X_train_withoutUsers **

# In[29]:

X_train_withoutUsersArray=np.array(X_train_withoutUsers)
X_test_withoutUsersArray=np.array(X_test_withoutUsers)


# ** Run model **

# In[30]:

model = TFFMRegressor(
    order=2,
    rank=7,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
    n_epochs=30,
    batch_size=-1,
    init_std=0.001,
    input_type='dense'
)


# In[31]:

model.fit(X_train_withoutUsersArray, y_tr, show_progress=True)
predictions = model.predict(X_test_withoutUsersArray)
print('MSE: {}'.format(mean_squared_error(y_te, predictions)))


# ## Make predictions:
# 
# (this is the messy part - very difficult to predict new movies using such a sparse array)

# Checking out how many unique users there are:

# In[32]:

ratings._userid.unique()


# Create DataFrame consisting only of user 125794:

# In[33]:

test_df_125794=pd.DataFrame(X[X['userid2']==125794])


# In[34]:

test_df_125794.drop('userid2', axis=1, inplace=True)


# Create list of unwatched movies for user125794:

# In[35]:

filtered=test_df_125794.filter(regex="movie.*")


# In[36]:

unwatched_user125794=list(filtered.columns[(filtered == 0).all()])


# Create DataFrame consisting only of user 82418: 

# In[48]:

test_df_82418=pd.DataFrame(X[X['userid2']==82418])


# In[49]:

test_df_82418.drop('userid2', axis=1, inplace=True)


# Create list of unwatched movies for user 82418:

# In[50]:

filtered=test_df_82418.filter(regex="movie.*")


# In[51]:

unwatched_user82418=list(filtered.columns[(filtered == 0).all()])


# Create dataframe consisting only of user 118205 (one of the most active users):

# In[37]:

test_df_118205=pd.DataFrame(X[X['userid2']==118205])


# In[38]:

test_df_118205.drop(['userid2'], axis=1, inplace=True)


# In[39]:

test_df_118205.head()


# User 118205 ---> at postition [-4] in arrays
# 
# User 125794 ---> at position [-2] in arrays ==== no good with 118205
# 
# User 82418 ---> at position [-5] in the arrays === no good with 118205
# 
# User 125794 with 82418 ----> [-2] needs to be turned off, [-5] turned on

# In[40]:


#temporary=np.array(test_df_118205.loc[test_df_118205['movieid_3']==1])


# In[ ]:




# ** Creating dictionary to map movie id back to title: **

# In[41]:

dictionary2=dict(zip(genres2.movieid,genres2.movienm))


# In[60]:

predictList=[]
moviesList=[]

for i in range(1,1000):
    ###create numpy matrix using movieid:
    temp1=np.array(test_df_125794.loc[test_df_125794[unwatched_user82418[i]]==1])
    if len(temp1) != 0:
        for each in temp1:
            each[-2]=0
            each[-5]=1
        pred=model.predict(temp1)
        predictList.append(np.average(pred))
        moviesList.append(int(unwatched_user82418[i].split('_')[1]))


# In[61]:

sanitycheck=list(zip(predictList,moviesList))
sanitycheck.sort(reverse=True)


# In[62]:

sanitycheck


# In[64]:

for i in range(0,50):
    print(dictionary2[sanitycheck[i][1]])


# In[ ]:




# In[ ]:



