import numpy as np
import pandas as pd

IN_FILE = 'data/events_train.csv'
OUT_FILE = 'data/events_part.csv'
CHUNK_SIZE = 10 ** 6
ENTRIES_FROM_CHUNK = 1000

try:
  result = pd.DataFrame()
  for partial_data in pd.read_csv(IN_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    partial_result = partial_data.sample(n=ENTRIES_FROM_CHUNK)
    result = result.append(partial_result)

  result.to_csv(OUT_FILE, index=False)
except:
  print('error - check input parameters')
