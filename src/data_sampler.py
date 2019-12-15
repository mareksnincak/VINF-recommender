import numpy as np
import pandas as pd

IN_FILE = 'data/merged/merged.csv'
OUT_FILE = 'data/recent/recent_merged_2w.csv'
CHUNK_SIZE = 10 ** 6
FROM_DATE = '2019-08-16'

try:
  result = pd.DataFrame()
  for partial_data in pd.read_csv(IN_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    filtered = partial_data[partial_data['timestamp'] >= FROM_DATE]
    result = result.append(filtered)

  result.to_csv(OUT_FILE, index=False)
except:
  print('error - check input parameters')
