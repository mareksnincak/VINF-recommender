import numpy as np
import pandas as pd

IN_FILE = 'data/events_part.csv'
FIELDS_TO_ANALYZE = 'all'

try:
  pd.set_option('display.max_columns', None)
  data = pd.read_csv(IN_FILE)
  print(data.isnull().sum())
  print(data.describe(include=FIELDS_TO_ANALYZE))
except:
  print('error - check input parameters')
