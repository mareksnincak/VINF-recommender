# evaluation - Precision,  Precision@k, nDCG
import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

# recommeder params
PURCHASE_WEIGHT = 3
MIN_UNIQUE_INTERACTIONS = 5 # around 8
NUMBER_OF_COMPONENTS = 150 # seems like the best results are somewhere around 50-60 range
RECOMMENDATION_COUNT = 10
BASE_WEIGHT = 3
MAX_WEIGHT = 3.5

EXACT_INTERACTIONS_COUNT = 5 # has to be >= MIN_UNIQUE_INTERACTIONS, just for testing

def searchByKey(lst, val):
  for key, value in lst.items(): 
    if val == value: 
      return key

class Recommender:
  def __init__(self, filename, test = False, test_size = 1000):
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
      data, self.test_data = train_test_split(data.sort_values(by=['timestamp']), shuffle = False, test_size=test_size)

    # most popular
    self.most_popular = data\
      .groupby(['customer_id', 'product_id'])\
      .agg({'customer_id': 'first', 'product_id': 'first'})\
      .reset_index(drop=True)
    self.most_popular = self.most_popular['product_id'].value_counts().head(RECOMMENDATION_COUNT).keys().tolist()

     # filter active only
    customer_activity = data.groupby(['customer_id'])['product_id'].nunique()
    # save user activity
    if test and EXACT_INTERACTIONS_COUNT:
      self.customer_activity = customer_activity
    active_customers = customer_activity[customer_activity >= MIN_UNIQUE_INTERACTIONS]
    data = data[data['customer_id'].isin(active_customers.index)]

    # assign weights to different events
    data['weight'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})

    # calculate weight for unique user-item interactions
    data = data\
      .groupby(['customer_id', 'product_id'])\
      .agg({'customer_id': 'first', 'product_id': 'first', 'weight': sum})\
      .reset_index(drop=True)
    data['weight'] = data['weight'].map(lambda x: min(BASE_WEIGHT +  2 * math.log10(x), MAX_WEIGHT))

    # create indexes for matrix
    self.user_mapping = {key: i for i, key in enumerate(data['customer_id'].unique())}
    self.item_mapping = {key: i for i, key in enumerate(data['product_id'].unique())}

    data['customer_id'] = data['customer_id'].map(lambda x: self.user_mapping[x])
    data['product_id'] = data['product_id'].map(lambda x: self.item_mapping[x])

    self.matrix = csr_matrix(
      (data['weight'], (data['customer_id'], data['product_id'])),
      shape=(len(self.user_mapping), len(self.item_mapping))
    )

    # matrix factorization
    import implicit
    model = implicit.als.AlternatingLeastSquares(factors=NUMBER_OF_COMPONENTS)
    model.fit(self.matrix, show_progress=True)

    self.user_matrix = model.item_factors
    self.item_matrix = np.transpose(model.user_factors)

    """ 
    # matrix factorization
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=NUMBER_OF_COMPONENTS)
    self.user_matrix = nmf.fit_transform(self.matrix)
    self.item_matrix = nmf.components_

    print(self.user_matrix @ self.item_matrix) """

    print('initialized')


  def recommend(self, user_id):
    # map id to index
    try:
      index = self.user_mapping[user_id]
    except:
      return self.most_popular

    recommendations = list(enumerate(self.user_matrix[index] @ self.item_matrix))
    recommendations.sort(key = lambda x: x[1], reverse = True)

    # get array of indexes of items that user already bought
    user_interactions = self.matrix[index].nonzero()[1]
    """ 
    i = list(map(lambda x: searchByKey(self.item_mapping, x), user_interactions))
    print(i) 
    """

    # get indexes of top RECOMMENDATION_COUNT items that are new for user
    recommendations = recommendations[:RECOMMENDATION_COUNT + len(user_interactions)]
    recommendations = list(filter(lambda x: x[0] not in user_interactions, recommendations))[:RECOMMENDATION_COUNT]

    # get item names
    recommendations = list(map(lambda x: searchByKey(self.item_mapping, x[0]), recommendations))
    return recommendations

  def test(self, user_count):
    print('Average precision@k and dcg@k test')
    precision_sum = 0
    dcg_sum = 0
    predictions = 0
    user_ids = np.unique(self.test_data['customer_id'])
    for uid in user_ids:
      if predictions == user_count:
        break

      # skip ones without ranking (they would get most popular items)
      try:
        self.user_mapping[uid]
      except:
        continue

      if EXACT_INTERACTIONS_COUNT:
        if self.customer_activity[uid] < EXACT_INTERACTIONS_COUNT:
          continue
        # print(self.customer_activity[uid])

      predictions += 1
      recommendations = self.recommend(uid)

      user_interactions = self.test_data[self.test_data['customer_id'] == uid]
      user_interactions = np.unique(user_interactions['product_id']).tolist()

      hits = 0
      index = 0
      dcg = 0
      for r in recommendations:
        index += 1
        if r in user_interactions:
          hits += 1
          if index == 1:
            dcg += 1
            continue
          dcg += 1 / math.log(index)
          
      dcg_sum = dcg_sum + dcg
      precision_sum = precision_sum + (hits / len(recommendations))

    print('Number of predictions:', predictions)
    if not predictions:
      return
    print('Average precision@k:', precision_sum / predictions)
    print('Average dcg@k:', dcg_sum / predictions)
    return
