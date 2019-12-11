import numpy as np
import pandas as pd

user_ids = pd.read_csv('challenge/vi_challenge_uID.csv', header=None)[0]
data = pd.read_csv('data/merged/merged.csv', dtype={
      'customer_id': object,
      'timestamp': object,
      'event_type': object,
      'product_id': object,
      'title': object,
      'category_name': object,
      'price': float,
    })

customer_activity = data.groupby(['customer_id'])['product_id'].nunique()

activity = [0] * 11
for uid in user_ids:
  try:
    if customer_activity[uid] >= 10:
      activity[10] += 1
      continue
    activity[customer_activity[uid]] += 1

  except:
    activity[0] += 1
    continue

print(activity)