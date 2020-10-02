import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("real estate.csv")

df['date']  = df['X1 transaction date']
df['age']  = df['X2 house age']
df['station']  = df['X3 distance to the nearest MRT station']
df['stores']  = df['X4 number of convenience stores']
df['price']  = df['Y house price of unit area']

df = df[['No','date','age' ,'station' ,'stores' , 'price']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

df['age'] = df['age'].round()
df['station'] = df['station'].round()
df['stores'] = df['stores'].round()

from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics, svm


x = df[['age', 'stores']]
y = df['price']


lab_enc = preprocessing.LabelEncoder()
y_enc = lab_enc.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_enc, test_size=0.3)

tr = LinearRegression()
tr.fit(x_train, y_train)
y_pred = tr.predict(x_test)

pickle.dump(tr, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))



print(model.predict([[5, 10]]))

















