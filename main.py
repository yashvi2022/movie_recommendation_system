import pandas as pd
import numpy as np
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# url ='http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# # response = requests.get(url, stream=True)

# urllib.request.urlretrieve(url, 'ml-latest-small.zip')

# print("File Downloaded")

# with zipfile.ZipFile('ml-latest-small.zip', 'r') as data:
#     data.extractall()

ratings = pd.read_csv('ml-latest-small/ratings.csv', usecols=['userId','movieId','rating'])  
movies = pd.read_csv ('ml-latest-small/movies.csv', usecols=['movieId','title'])

print("Dataframes loaded succesfully!")
data = pd.merge(ratings, movies, on= 'movieId')

user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix.fillna(0, inplace=True)
print (user_item_matrix)

user_similarity = cosine_similarity(user_item_matrix)
print("User Similarity", user_similarity)

def find_similar_users(user_id, user_similarity, top_n= 5):
    user_index = user_item_matrix.index.get_loc(user_id)
    # print("User Index", user_index)
    similar_users = user_similarity[user_index]
    print("Similar Users", similar_users)
    similar_users_indices = np.argsort(similar_users)[::-1][1:top_n+1]
    print("Similar User Indices", similar_users_indices)
    return user_item_matrix.index[similar_users_indices]

#_similar_users(2, user_similarity)

def generate_recommendations(user_id, user_similarity, user_item_matrix, top_n=5):
    similar_users = find_similar_users(user_id, user_similarity)
    similar_users_ratings = user_item_matrix.loc[similar_users]
    average_ratings = similar_users_ratings.mean()
    recommended_movies = average_ratings.sort_values(ascending=False).head(top_n)
    
    return recommended_movies

recommendations = generate_recommendations(4, user_similarity, user_item_matrix)

def evaluate_model(user_id, user_similarity, user_item_matrix, actual_ratings, top_n = 5):
    recommendations = generate_recommendations(user_id, user_similarity, user_item_matrix, top_n)
    common_movies = recommendations.index.intersection(actual_ratings.index)
    precision = len(common_movies) / top_n
    recall = len(common_movies) / len(actual_ratings[actual_ratings > 0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score


actual_ratings = user_item_matrix.loc[4]  
precision, recall, f1_score = evaluate_model(4, user_similarity, user_item_matrix, actual_ratings)
print(f"{precision=}")
print(f"{recall=}")
print(f"{f1_score=}")