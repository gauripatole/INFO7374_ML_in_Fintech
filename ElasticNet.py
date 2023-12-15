import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import math



# define dataset
start_date = datetime(2020,1,1)
end_date = datetime(2023,12,31)
DATA_raw = pd.read_csv('INFOP7374_FeatureMart4GS.csv', index_col = [0])
DATA_raw.describe()
# copy the data
DATA = DATA_raw.copy() 
DATA['Date'] = pd.to_datetime(DATA['Date'])
# X and Y have to use the same index and index must be of the same data type



##Nomalize the Volumn attribute in the model
#for column in DATA_raw.columns[1:7]:
 #   DATA[column] = (DATA_raw[column] -
  #                         DATA_raw[column].mean()) / DATA_raw[column].std()  

#check null values
#STOCK_raw = yf.download('LULU', start_date, end_date)
#STOCK_raw.describe()
##STOCK = STOCK_raw.copy()

indicatorCol = ['USRECDP', 'USRECDM', 'USRECD']
for column in DATA_raw.columns[1:]:
    if column not in indicatorCol:
        DATA[column] = (DATA_raw[column] - DATA_raw[column].mean()) / DATA_raw[column].std()
    else:
        continue
        #DATA[column] = DATA_raw[column].astype(int)
df1 = DATA.iloc[:,6:22]
df2 = DATA.iloc[:,25:30]
df_concat = pd.concat([df1,df2], axis=1)
#X = df_concat.set_index('Date').fillna(method='bfill')
X = df_concat.fillna(method='bfill')
#exclude_columns = ['Adj Close', 'Yest_Close', 'stock_return']
y = np.array(DATA['Adj Close']).reshape(-1,1)
#y = np.diff(np.log(DATA['Adj Close'].values))
#y = np.append(y[0], y)
X = sm.add_constant(X)



#train,test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = 40)
##
##
####Feature Selection using Elastic Net
a=0.01
##
##
##
####Tune the parameters for the model
###model = ElasticNet(alpha=a, l1_ratio=0.4)
model_prep = ElasticNet(alpha=a, fit_intercept=False, l1_ratio=0.5).fit(X_train, y_train)
model_select = X_train.columns[np.abs(model_prep.coef_)!=0.0]
x_train = X_train[model_select]
model = sm.OLS(y_train,x_train).fit()
print(model.summary())
##y_pred = model.predict(x_train)
##corr_model = ss.pearsonr(y_pred, y_train)[0]
##print('model Elastic Net: corr (Y, Y_pred) = '+str(corr_model))
##print('ElasticNet selected ' +str(len(model_select)) +' features: ', model_select.values)
##
##
##
y_hat = model.predict(X_test[model_select])
####RMSE
MSE = mean_squared_error(y_test, y_hat)
RMSE = math.sqrt(MSE)
print('Elastic Net')
print('------------------------\n')
print('Root mean square error is       \n',RMSE)
##
# visualize feature importance (run each line sequentially)
coeff = model.params
plt.figure(figsize=(10,6))
plt.barh(list(x_train.columns), coeff)
plt.xlabel("Feature Importance (Coefficient)")
plt.ylabel("Features")
plt.title("Feature Importance in Linear Regression based on Elastic Net for LULU")

### (1) bar chart
##df_feature_importance.plot(kind='bar');
### (2) box plot
##sns.boxplot(x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
### (3) strip plot
##sns.stripplot(x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
### (4) swarm plot
##sns.swarmplot(x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
### (5) all above in one plot
##fig, axes = plt.subplots(4, 1, figsize=(16, 8))
##df_feature_importance.plot(kind='bar', ax=axes[0], title='Plots Comparison for Feature Importance');
##sns.boxplot(ax=axes[1], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
##sns.stripplot(ax=axes[2], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
##sns.swarmplot(ax=axes[3], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
plt.tight_layout()

