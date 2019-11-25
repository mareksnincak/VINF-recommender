import numpy as np
import pandas as pd
import math

IN_FILE = 'data/recent/recent_merged_hour.csv'
OUT_FILE = 'data/dataset.csv'
PURCHASE_WEIGHT = 3
MIN_INTERACTIONS = 5

try:
  data = pd.read_csv(IN_FILE)
  data['weight'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})

  data = data.groupby(['customer_id', 'title']).agg({'customer_id': 'first', 'title': 'first', 'weight': sum})

  data['weight'] = data['weight'].apply(lambda x: 1 + math.log(x))
  data.to_csv(OUT_FILE, index=False)
except:
  print('error - check input parameters')
