import numpy as np
import pandas as pd

IN_FILE = 'data/dataset.csv'
ANALYZE_BY_FIELDS = False

try:
  pd.set_option('display.max_columns', None)

  if ANALYZE_BY_FIELDS:
    fields = pd.read_csv(IN_FILE, nrows=1).columns.tolist()
    print(fields)

    for field in fields:
      data = pd.read_csv(IN_FILE, usecols=[field])
      
      if field == 'timestamp':
        print('timestamp analysis: ')
        print(min(data['timestamp']))
        print(max(data['timestamp']))

      print(data.isnull().sum())
      print(data.describe(include='all'))
      print(data.dtypes)

  else:
      data = pd.read_csv(IN_FILE)
      print('timestamp analysis: ')
      print(min(data['timestamp']))
      print(max(data['timestamp']))

      print(data.isnull().sum())
      print(data.describe(include='all'))
      print(data.dtypes)

      print('average user interactions per item: ')
      print(data.groupby(['title'])['customer_id'].count().mean())
except:
  print('error - check input parameters')
