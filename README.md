# Movie-TV-Show-Recommender-System
Content Based Recommender System 
------------------------------------------------------ 
This recommender system is built on a netflix dataset taken from kaggle. 'https://www.kaggle.com/shivamb/netflix-shows'  Content based filtering approach is taken here, and the core of everything here are the CountVectorizer and CosineSimilarity functions in python library SkLearn.   Basically, the CountVectorizer takes all the rows of the selected features, converts them into a count_matrix, then uses the cosine similarity method to find out the similarities between the desired movie/TVshow and all the other rows.  The output given by the CosineSimilarity function is then sorted in descending order so the most similar movies are displayed first, then the  top 10 movies are displayed by accessing the first 10 records and referencing the index numbers associated with each similarity score.  

Libraries used- 
Pandas 
Numpy 
NLTK 
String 
SKLearn 
Itertools
