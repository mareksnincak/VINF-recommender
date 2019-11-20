import numpy as np
import pandas as pd

IN_FILE = 'data/dataset.csv'
FIELDS_TO_ANALYZE = pd.read_csv(IN_FILE, nrows=1).columns.tolist()

try:
  pd.set_option('display.max_columns', None)
  for field in FIELDS_TO_ANALYZE:
    data = pd.read_csv(IN_FILE, usecols=[field])
    
    if field == 'timestamp':
      print('timestamp_analysis: ')
      print(min(data['timestamp']))
      print(max(data['timestamp']))

    print(data.isnull().sum())
    print(data.describe(include='all'))
    print(data.dtypes)
except:
  print('error - check input parameters')
