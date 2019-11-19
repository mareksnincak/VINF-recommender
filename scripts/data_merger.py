import numpy as np
import pandas as pd
import os

data = pd.read_csv('data/purchases_train.csv')
data.to_csv('data/purchases_train_stripped.csv', index=False, header=False)

os.chdir('C:/Users/msnin/Desktop/git/VINF-recommender/data')
os.system('cmd /c "copy events_train.csv+purchases_train_stripped.csv merged.csv"')
