# Movie-Recommendation-System
It is a movie recommendation system built on Jupyter notebook and Python


Pre-requisite libraries :- pandas, numpy, sklearn, rake_nltk, ast

Dataset link :- https://drive.google.com/file/d/1bzMdhDaSnk_A-yyN9cwnKdam-4GM3tK4/view?usp=sharing

Functions used :-
literal_eval - It is used to evaluate some columns of dataset which contained python objects such as lists and dictionaries.

get_director - It is only used to extract the name of director of each movie from dataset. If director not present, return a null string.

get_list - It is used to make a list of cast, keywords, genres and production companies from the dictionaries of their columns.
	   (I used all the cast names available for all movies since cast is an important feature and movies like 'The Avengers' had more
	    than 3-5 important cast members.)

clean_data - It is used to convert all the features present in features list to lower strings and remove the spaces between first name and
	     last name of the cast so that the model does not confuse to have any similarity between Chris Evans and Chris Hemsworth.
	     Eg. - 'Natalie Portman' is converted to 'natalieportman'.

create_soup - It is used to join all the remaining words after removing stopwords using rake_nltk to crate soup which can be seen in file.

get_recommendations - It is used to get the recommended movie names by letting the user provide a movie name.
