import numpy as np 
import pandas as pd 
# Import graphical plotting libraries 
import matplotlib.pyplot as plt 
import seaborn as sns
# Import Linear Regression Machine Learning Libraries
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

data = pd.read_csv(r"D:\FSDS Material\Dataset\L-R-R.csv")

data = data.drop(['car_name'], axis = 1)
data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
data = pd.get_dummies(data,columns = ['origin'])
data = data.replace('?', np.nan)
data = data.apply(lambda x: x.fillna(x.median()), axis = 0)

# Check and drop 'car_name' column if it exists
if 'car_name' in data.columns:
    data = data.drop(['car_name'], axis=1)

# Replace numeric 'origin' values with region names
data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})

# Perform one-hot encoding on 'origin'
data = pd.get_dummies(data, columns=['origin'])

# Replace '?' with NaN
data = data.replace('?', np.nan)

# Convert all numeric columns to their proper types
data = data.apply(pd.to_numeric, errors='ignore')

# Fill missing values with median only for numeric columns
data = data.apply(lambda x: x.fillna(x.median()) if x.dtype.kind in 'biufc' else x, axis=0)