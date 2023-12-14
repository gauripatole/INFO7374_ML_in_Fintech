import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import math


##X,y
df = pd.read_csv("INFOP7374_FeatureMart4GS.csv")
data = df.values
y = data[:, 6]
X = data[:, 8:29]

#train,test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

##Nomalize the Volumn attribute in the model


##Model
#model = ElasticNet(alpha=0.4, l1_ratio=0.4)
#model.fit(X_train,y_train)


#y_hat = model.predict(X_test)s
##RMSE
#MSE = mean_squared_error(y_test, y_hat)
#RMSE = math.sqrt(MSE)
print('Elastic Net')
print('------------------------\n')
print('Root mean square error is       \n',rmse)
