import numpy as np
import pandas as pd

IN_FILE = 'data/merged/merged.csv'
ANALYZE_BY_FIELDS = False

pd.set_option('display.max_columns', None)

if ANALYZE_BY_FIELDS:
  fields = pd.read_csv(IN_FILE, nrows=1).columns.tolist()
  print(fields)

  for field in fields:
    data = pd.read_csv(IN_FILE, usecols=[field], low_memory=False)
    
    if field == 'timestamp':
      print('timestamp analysis: ')
      print(min(data['timestamp']))
      print(max(data['timestamp']))

    print(data.isnull().sum())
    print(data.describe(include='all'))
    print(data.dtypes)

else:
    data = pd.read_csv(IN_FILE, dtype={
      'customer_id': object,
      'timestamp': object,
      'event_type': object,
      'product_id': object,
      'title': object,
      'category_name': object,
      'price': float,
    })
    print('timestamp analysis: ')
    print(min(data['timestamp'].dropna()))
    print(max(data['timestamp'].dropna()))

    print(data.isnull().sum())
    print(data.describe(include='all'))
    print(data.dtypes)

    print('average user interactions per item: ')
    print(data.groupby(['customer_id'])['product_id'].nunique().mean())
