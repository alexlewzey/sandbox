import time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from haversine import haversine, Unit

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

DATA_TAXI = r'C:\Users\alewz\Google Drive\programming\projects_al\first_pytourch\PYTORCH_NOTEBOOKS\Data\NYCTaxiFares.csv'

taxi = pd.read_csv(DATA_TAXI)
print(taxi.head())
print(taxi.dtypes)

taxi['pickup'] = list(zip(taxi['pickup_longitude'], taxi['pickup_latitude']))
taxi['dropoff'] = list(zip(taxi['dropoff_longitude'], taxi['dropoff_latitude']))

vhaversine = np.vectorize(haversine)

start_time = time.time()
taxi['distance'] = vhaversine(taxi.pickup.values, taxi.dropoff.values, unit=Unit.MILES)
time_taken_vectorised = time.time() - start_time

start_time = time.time()
taxi['distance_apply'] = taxi.apply(lambda row: haversine(row['pickup'], row['dropoff'], Unit.MILES), axis=1)
time_taken_apply = time.time() - start_time

print(f'time_taken_vectorised: {time_taken_vectorised}')
print(f'time_taken_apply: {time_taken_apply}')
print(f'vectorised runs {round((time_taken_apply / time_taken_vectorised), 2)} times faster than apply ie apply sucks')
