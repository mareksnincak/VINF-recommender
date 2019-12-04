# evaluation - Precision,  Precision@k, nDCG
import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

IN_FILE = 'data/recent/recent_merged_hour.csv'
PURCHASE_WEIGHT = 3
MIN_INTERACTIONS = 5

def searchByKey(lst, val):
  for key, value in lst.items(): 
         if val == value: 
             return key

class Recommender:
  def __init__(self):
    data = pd.read_csv(IN_FILE, dtype={
      'customer_id': object,
      'timestamp': object,
      'event_type': object,
      'product_id': object,
      'title': object,
      'category_name': object,
      'price': float,
    })

    # test train split
    # self.__train, self.__test = train_test_split(data.sort_values(by=['timestamp']), test_size=0.2)

    # filter active only
    customer_activity = data.groupby(['customer_id'])['title'].nunique()
    active_customers = customer_activity[customer_activity >= MIN_INTERACTIONS]
    data = data[data['customer_id'].isin(active_customers.index)]

    # assign weights to different events
    data['weight'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})

    # calculate weight for unique user-item interactions
    data = data\
      .groupby(['customer_id', 'title'])\
      .agg({'customer_id': 'first', 'title': 'first', 'weight': sum})\
      .reset_index(drop=True)
    data['weight'] = data['weight'].map(lambda x: 5 +  math.log10(x))

    # create indexes for matrix
    self.user_mapping = {key: i for i, key in enumerate(data['customer_id'].unique())}
    self.item_mapping = {key: i for i, key in enumerate(data['title'].unique())}

    data['customer_id'] = data['customer_id'].map(lambda x: self.user_mapping[x])
    data['title'] = data['title'].map(lambda x: self.item_mapping[x])

    self.matrix = csr_matrix(
      (data['weight'], (data['customer_id'], data['title'])),
      shape=(len(self.user_mapping), len(self.item_mapping))
    )

    nmf = NMF(n_components=50)
    self.user_matrix = nmf.fit_transform(self.matrix)
    self.item_matrix = nmf.components_


  def recommend(self, user_id):
    try:
      index = int(user_id) # should use self.user_mapping to get index from user_id in future
      ratings = list(enumerate(self.user_matrix[index] @ self.item_matrix))
      ratings.sort(key = lambda x: x[1], reverse = True)

      # get array of indexes of items that user already bought
      user_interactions = self.matrix[index].nonzero()[1]

      # get indexes of top 10 items that are new for user
      ratings = ratings[:10 + len(user_interactions)]
      ratings = list(filter(lambda x: x[0] not in user_interactions, ratings))[:10]

      # get item names
      ratings = list(map(lambda x: searchByKey(self.item_mapping, x[0]), ratings))
      return ratings
    except:
      print('bad input')
      return


r = Recommender()

while True:
  user_input = input('enter user id: ')
  if user_input == 'q':
    break

  print(r.recommend(user_input))
