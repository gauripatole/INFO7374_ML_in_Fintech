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


##X,y
data = pd.read_csv("INFOP7374_FeatureMart4GS.csv")

##Model
#model = ElasticNet(alpha=0.4, l1_ratio=0.4)
#model.fit(X,y)

##RMSE
