# evaluation - Precision,  Precision@k, nDCG
import numpy as np
import pandas as pd
import math

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

IN_FILE = 'data/recent/recent_merged_hour.csv'
PURCHASE_WEIGHT = 3
MIN_INTERACTIONS = 3

data = pd.read_csv(IN_FILE)

# test train split
train, test = train_test_split(data.sort_values(by=['timestamp']), test_size=0.2)

# filter active only
customer_activity = data.groupby(['customer_id'])['title'].nunique()
active_customers = customer_activity[customer_activity >= MIN_INTERACTIONS]
data = data[data['customer_id'].isin(active_customers.index)]
exit()

# assign weights to different events
data['weight'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})

# calculate weight for unique user-item interactions
data = data.groupby(['customer_id', 'title']).agg({'customer_id': 'first', 'title': 'first', 'weight': sum}).reset_index(drop=True)
data['weight'] = data['weight'].apply(lambda x: 1 + math.log(x))

item_matrix = pd.pivot_table(data, values='weight', index='title', columns='customer_id')

print(data.head())