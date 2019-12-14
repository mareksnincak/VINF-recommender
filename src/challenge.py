import numpy as np
import pandas as pd

from recommender import Recommender

IN_FILE = 'data/recent/recent_merged.csv'

r = Recommender(IN_FILE)

user_ids = pd.read_csv('challenge/vi_challenge_uID.csv', header=None)[0]

rows = []
for uid in user_ids:
  recommendations = r.recommend(uid)
  for item in recommendations:
    row = [uid, item]
    rows.append(row)

result = pd.DataFrame(rows, columns=['customer_id', 'product_id'])
result.to_csv('challenge/submission.csv', index=False)