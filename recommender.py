#Importing all the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain

## Reading the dataset

df = pd.read_csv('netflix_titles.csv')

###### There appears to be some null values in the columns 'Director', 'Cast', 'Country' and 'date_added'. We will replace them by a whitespace.

df.fillna('', inplace=True)

df.isnull().sum()

##### The features that i'm choosing are ['type', 'title', 'director', 'cast', 'listed_in', 'description'].

###### First taking out unnecessary 'stopwords' from the column 'description'

def cleanup(text):
    #removing punctuations
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    #removing the stopwords
    clean_info = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    return clean_info

df['description'] = df['description'].apply(cleanup)

###### Now we're combining all the relevant features and storing them in a single column.

def features(row):
    return row['type']+" "+row['title']+" "+row['director']+' '+row['cast']+' '+row['listed_in']+' '+str(row['description'])

df['combined_features'] = df.apply(features, axis=1)

###### Using the CountVectorizer function on the combined_features column to get a 'count_matrix' that contains the word counts of each row

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

###### Now calculating the cosine similarity of all the rows.
#The resulting matrix of the cosine similarities will be stored in the 'cos_sim' variable

cos_sim = cosine_similarity(count_matrix)

###### Functions to get the Movie name from it's index value (get_title_from_index) and a function to get index from the title (get_index_from_title)

def get_index_from_title(name):
    return df.index[df.title==name].values
def get_title_from_index(m_index):
    return df[df.index==m_index]['title'].values


def suggest(name):

    movie_index = get_index_from_title(name) #getting the index of the movie from the movie name
    similar_movies = cos_sim[movie_index] #fetching the similarity scores of all the movies against the given 'movie_name' from the 'cosine_similarity' matrix
    similar_movies = list(enumerate(chain.from_iterable(similar_movies))) #Converting the resulting list of similarity scores into a list of tuples where each tuple contains the individual movie's index on the 0th index and it's similarity score on the 1st index
    sorted_movies = sorted(similar_movies, key= lambda x: x[1], reverse=True) #Now using the similarity score of each tuple to arrange the similarity scores in descending order so the most similar movies come first
    ###### Displaying the recommendation using the index from each tuple associated with the similarity score and fetching the movie name using that index as a reference in the original df
    #displaying the top 10 suggestions
    print("Based on ", name, ", you might like:")
    i=0
    for movie in sorted_movies:
        print(str(get_title_from_index(movie[0])))
        i+=1
        if i>10:
            break
#Asking the user for the movie/show based on which the suggestions will be produced

m_name = input("What was the last movie or TV Show you watched?: ") #try: Narcos, Stranger Things, Friends, Little Things, Special 26, Naam Shabana, Raees
suggest(m_name)
