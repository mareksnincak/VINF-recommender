import numpy as np
import pandas as pd

IN_FILE = 'data/recent/recent_merged.csv'
OUT_FILE = 'data/dataset.csv'
PURCHASE_WEIGHT = 5

try:
  data = pd.read_csv(IN_FILE)
  data['event_type'] = data['event_type'].map({'view_item': 1, 'purchase_item': PURCHASE_WEIGHT})
  data.to_csv(OUT_FILE, index=False)
except:
  print('error - check input parameters')
