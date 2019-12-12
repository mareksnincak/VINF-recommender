import numpy as np
import pandas as pd

from recommender import Recommender

IN_FILE = 'data/recent/recent_merged.csv'

# testing params
TEST = True
TEST_USER_COUNT = 100
TEST_SIZE = 1 / 7 # 1 / 60 - adjust so data in test sample are from around one day

r = Recommender(IN_FILE, TEST, TEST_SIZE)

if TEST:
  r.test(TEST_USER_COUNT)
  quit()

while True:
  user_input = input('enter user id: ')
  if user_input == 'q':
    break

  print(r.recommend(user_input))
