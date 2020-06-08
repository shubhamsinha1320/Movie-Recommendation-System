## Download link for the dataset (.csv file):
## https://drive.google.com/file/d/1bzMdhDaSnk_A-yyN9cwnKdam-4GM3tK4/view?usp=sharing

import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('tmdb_5000_movies.csv', delimiter=',')
print(df.head())

print(df.info())

print(df.shape)

from ast import literal_eval

# Specifying the desired features list and applying literal_eval function to it
features = ['cast', 'crew', 'keywords', 'genres', 'production_companies']

for feature in features:
    df[feature] = df[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job']=='Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names
    return [] # returning empty list incase of missing or malformed data

df['director'] = df['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres', 'production_companies']

for feature in features:
    df[feature] = df[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
    # checking whether director exists. If not, returning empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ",""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres', 'production_companies']

for feature in features:
    df[feature] = df[feature].apply(clean_data)

print(df[['title', 'cast', 'director', 'keywords', 'genres', 'production_companies']].head(10))

Overview_Keywords = []

for index, row in df.iterrows():
    overview = row['overview']
    
    # instantiating Rake, by default uses english stopwords from NLTK
    # and discard all puntuation characters
    r = Rake()

    # extracting the words by passing the text
    r.extract_keywords_from_text(overview)

    # getting the dictionary whith key words and their scores
    key_words_dict_scores = r.get_word_degrees()
    
    Overview_Keywords.append(list(key_words_dict_scores.keys()))
    
df['Overview_Keywords'] = Overview_Keywords

# dropping the overview column
df.drop(columns = ['overview'], inplace = True)

df.set_index('title', inplace = True)
print(df[['cast', 'director', 'keywords', 'genres', 'production_companies', 'Overview_Keywords']].head())

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(x['production_companies']) + ' ' + ' '.join(x['Overview_Keywords'])
df['soup'] = df.apply(create_soup, axis=1)

print(df[['soup']].head(10))

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['soup'])

# Cosine similarity is a method to measure the difference between two non-zero vectors of an inner product space
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# list used later to match the indexes
indices = pd.Series(df.index)
print(indices[:10])

# function that takes in movie title as input and returns the top 10 recommended movies
def get_recommendations(title, cosine_sim = cosine_sim):
    
    recommended_movies = []
    
    # getting the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies

print(get_recommendations('Battle: Los Angeles'))
