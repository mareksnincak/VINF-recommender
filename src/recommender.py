# evaluation - Precision,  Precision@k, nDCG
import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# recommeder params
PURCHASE_WEIGHT = 3
MIN_UNIQUE_INTERACTIONS = 10 # around 8
RECOMMENDATION_COUNT = 10
BASE_WEIGHT = 3
MAX_WEIGHT = 5
MOST_SIMILAR = 20

def searchByKey(lst, val):
  for key, value in lst.items(): 
    if val == value: 
      return key

class Recommender:
  def __init__(self, filename, test = False, test_size = 1 / 60):
    print('initializing recommender')
    data = pd.read_csv(filename, dtype={
      'customer_id': object,
      'timestamp': object,
      'event_type': object,
      'product_id': object,
      'title': object,
      'category_name': object,
      'price': float,
    })

    # test train split
    if test:
      data, self.testData = train_test_split(data.sort_values(by=['timestamp']), shuffle = False, test_size=test_size)

    # most popular
    self.most_popular = data\
      .groupby(['customer_id', 'product_id'])\
      .agg({'customer_id': 'first', 'product_id': 'first'})\
      .reset_index(drop=True)
    self.most_popular = self.most_popular['product_id'].value_counts().head(RECOMMENDATION_COUNT).keys().tolist()

    # filter active only
    customer_activity = data.groupby(['customer_id'])['product_id'].nunique()
    active_customers = customer_activity[customer_activity >= MIN_UNIQUE_INTERACTIONS]
    data = data[data['customer_id'].isin(active_customers.index)]

    # assign weights to different events
    data['weight'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})

    # calculate weight for unique user-item interactions
    data = data\
      .groupby(['customer_id', 'product_id'])\
      .agg({'customer_id': 'first', 'product_id': 'first', 'weight': sum})\
      .reset_index(drop=True)
    data['weight'] = data['weight'].map(lambda x: min(BASE_WEIGHT +  math.log(x), MAX_WEIGHT))

    # create indexes for matrix
    self.user_mapping = {key: i for i, key in enumerate(data['customer_id'].unique())}
    self.item_mapping = {key: i for i, key in enumerate(data['product_id'].unique())}

    data['customer_id'] = data['customer_id'].map(lambda x: self.user_mapping[x])
    data['product_id'] = data['product_id'].map(lambda x: self.item_mapping[x])

    self.matrix = csr_matrix(
      (data['weight'], (data['customer_id'], data['product_id'])),
      shape=(len(self.user_mapping), len(self.item_mapping))
    )

    """ print(self.matrix[:, 0])
    print(searchByKey(self.item_mapping, 0))
    while True:
      user_input = input('u: ')
      print(searchByKey(self.user_mapping, int(user_input))) """

    print('initialized')


  def recommend(self, user_id):
    # map id to index
    try:
      index = self.user_mapping[user_id]
    except:
      return self.most_popular

    # get k most similar users
    similarities = cosine_similarity(self.matrix[index], self.matrix)[0]
    similar_users = np.argsort(similarities)[::-1][:MOST_SIMILAR + 1]

    # calculate score as sum from their ratings divided by 1 - their similarity
    # similar_users = similar_users[1:]
    score = self.matrix[similar_users[1]].toarray()[0]
    for u in similar_users[1:]:
      if similarities[u] == 0 or similarities[u] >= 1:
        continue

      score = score + np.array(np.divide(self.matrix[u].toarray()[0], 1 - similarities[u]))

    # get array of indexes of items that user has already interacted with
    user_interactions = self.matrix[index].nonzero()[1]

    # get indexes of best RECOMMENDATION_COUNT scores of items new to user
    recommendations = score.argsort()[::-1][:RECOMMENDATION_COUNT + len(user_interactions)]
    recommendations = list(filter(lambda x: x not in user_interactions, recommendations))[:RECOMMENDATION_COUNT]
    """ print(recommendations)
    while True:
      user_input = input('p: ')
      if user_input == 'q':
        break
      print(searchByKey(self.item_mapping, int(user_input))) """

    # get item names
    recommendations = list(map(lambda x: searchByKey(self.item_mapping, x), recommendations))

    return recommendations

  def test(self, user_count):
    print('average precision@k test')
    precision_sum = 0
    predictions = 0
    user_ids = np.unique(self.testData['customer_id'])
    for uid in user_ids:
      if predictions == user_count:
        break
      # skip most popular
      try:
        self.user_mapping[uid]
      except:
        continue

      predictions += 1
      recommendations = self.recommend(uid)

      user_interactions = self.testData[self.testData['customer_id'] == uid]
      user_interactions = np.unique(user_interactions['product_id']).tolist()

      hits = 0
      for r in recommendations:
        if r in user_interactions:
          hits += 1

      precision_sum = precision_sum + (hits / len(recommendations)) 

    print(predictions)
    print(precision_sum / predictions)
    return
