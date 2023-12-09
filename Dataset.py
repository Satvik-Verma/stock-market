import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Apple-data-from-2013-till-Dec-2023.csv')

print(dataset.head())

print(dataset.isnull().sum())