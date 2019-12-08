# evaluation - Precision,  Precision@k, nDCG
import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

IN_FILE = 'data/recent/recent_merged.csv'
PURCHASE_WEIGHT = 3
MIN_INTERACTIONS = 5
NUMBER_OF_COMPONENTS = 50
TEST_USER_COUNT = 1000

def searchByKey(lst, val):
  for key, value in lst.items(): 
    if val == value: 
      return key

class Recommender:
  def __init__(self, testMode=0):
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
    if testMode:
      data, self.testData = train_test_split(data.sort_values(by=['timestamp']), test_size=0.2)

    # filter active only
    customer_activity = data.groupby(['customer_id'])['product_id'].nunique()
    active_customers = customer_activity[customer_activity >= MIN_INTERACTIONS]
    data = data[data['customer_id'].isin(active_customers.index)]

    # assign weights to different events
    data['weight'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})

    # calculate weight for unique user-item interactions
    data = data\
      .groupby(['customer_id', 'product_id'])\
      .agg({'customer_id': 'first', 'product_id': 'first', 'weight': sum})\
      .reset_index(drop=True)
    data['weight'] = data['weight'].map(lambda x: 5 +  math.log10(x))

    # create indexes for matrix
    self.user_mapping = {key: i for i, key in enumerate(data['customer_id'].unique())}
    self.item_mapping = {key: i for i, key in enumerate(data['product_id'].unique())}

    data['customer_id'] = data['customer_id'].map(lambda x: self.user_mapping[x])
    data['product_id'] = data['product_id'].map(lambda x: self.item_mapping[x])

    self.matrix = csr_matrix(
      (data['weight'], (data['customer_id'], data['product_id'])),
      shape=(len(self.user_mapping), len(self.item_mapping))
    )

    nmf = NMF(n_components=NUMBER_OF_COMPONENTS)
    self.user_matrix = nmf.fit_transform(self.matrix)
    self.item_matrix = nmf.components_


  def recommend(self, user_id):
    # map id to index
    try:
      index = self.user_mapping[user_id]
    except:
      # recommend most popular or something
      # print('new user')
      return

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

  def test(self):
    print('precision test')
    hits = 0
    predictions = 0

    user_ids = np.unique(self.testData['customer_id'])[:TEST_USER_COUNT]

    for uid in user_ids:
      recommendations = self.recommend(uid)
      if not recommendations:
        continue

      predictions += 1

      user_interactions = self.testData[self.testData['customer_id'] == uid]
      user_interactions = np.unique(user_interactions['product_id']).tolist()

      for r in recommendations:
        if r in user_interactions:
          hits += 1
          break

    print(hits / predictions)
    return


r = Recommender(1)
r.test()
while True:
  user_input = input('enter user id: ')
  if user_input == 'q':
    break

  # print(r.recommend(user_input))
