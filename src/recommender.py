# evaluation - Precision,  Precision@k, nDCG
import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

IN_FILE = 'data/merged/merged.csv'
PURCHASE_WEIGHT = 3
MIN_INTERACTIONS = 0

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
train, test = train_test_split(data.sort_values(by=['timestamp']), test_size=0.2)

# filter active only
customer_activity = data.groupby(['customer_id'])['title'].nunique()
active_customers = customer_activity[customer_activity >= MIN_INTERACTIONS]
data = data[data['customer_id'].isin(active_customers.index)]

# assign weights to different events
data['weight'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})

# calculate weight for unique user-item interactions
data = data.groupby(['customer_id', 'title']).agg({'customer_id': 'first', 'title': 'first', 'weight': sum}).reset_index(drop=True)
data['weight'] = data['weight'].map(lambda x: 100 + int(100 * math.log10(x)))

# create indexes for matrix
user_mapping = {key: i for i, key in enumerate(data['customer_id'].unique())}
item_mapping = {key: i for i, key in enumerate(data['title'].unique())}

data['customer_id'] = data['customer_id'].map(lambda x: user_mapping[x])
data['title'] = data['title'].map(lambda x: item_mapping[x])

matrix = csr_matrix((data['weight'], (data['customer_id'], data['title'])), shape=(len(user_mapping), len(item_mapping)))
print(matrix)
