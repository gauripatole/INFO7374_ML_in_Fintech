import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import math


##X,y, use the features
df = pd.read_csv("INFOP7374_FeatureMart4GS.csv")
data = df.values
y = data[:, 6]
X = data[:, 8:29]

##Nomalize the Volumn attribute in the model
for column in data.columns[1:]:
    STOCK[column] = (STOCK_raw[column] -
                           STOCK_raw[column].mean()) / STOCK_raw[column].std()  

#check null values

#train,test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)






##Feature Selection using Elastic Net
a=1



##Tune the parameters for the model
#model = ElasticNet(alpha=a, l1_ratio=0.4)
model_prep = linear_model.ElasticNet(alpha=a, fit_intercept=False).fit(X_train, y_train)
model_select = X.columns[np.abs(model_prep.coef_)!=0.0]
x = X[model_select]
model = sm.OLS(y_train,x).fit()
print(model.summary())
y_hat = model.predict(x)
corr_model4 = ss.pearsonr(y_pred_model4, y)[0]
print('model 4 Elastic Net: corr (Y, Y_pred) = '+str(corr_model3))
print('ElasticNet selected ' +str(len(model_select)) +' features: ', model_select.values)

#model.fit(X_train,y_train)


#y_hat = model.predict(X_test)s
##RMSE
#MSE = mean_squared_error(y_test, y_hat)
#RMSE = math.sqrt(MSE)
#print('Elastic Net')
#print('------------------------\n')
#print('Root mean square error is       \n',rmse)

# 1. average feature importance
df_feature_importance = pd.DataFrame(model.feature_importances_, index=X.columns, \
                                     columns=['feature importance']).sort_values('feature importance', ascending=False)
print(df_feature_importance)

# 2. all feature importance for each tree
df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in model.estimators_], columns=X.columns)
df_feature_all.head()
# Melted data i.e., long format
df_feature_long = pd.melt(df_feature_all,var_name='feature name', value_name='values')
print(df_feature_long.iloc[0:102])


# 3. visualize feature importance (run each line sequentially)
# (1) bar chart
df_feature_importance.plot(kind='bar');
# (2) box plot
sns.boxplot(x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
# (3) strip plot
sns.stripplot(x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
# (4) swarm plot
sns.swarmplot(x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
# (5) all above in one plot
fig, axes = plt.subplots(4, 1, figsize=(16, 8))
df_feature_importance.plot(kind='bar', ax=axes[0], title='Plots Comparison for Feature Importance');
sns.boxplot(ax=axes[1], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
sns.stripplot(ax=axes[2], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
sns.swarmplot(ax=axes[3], x="feature name", y="values", data=df_feature_long, order=df_feature_importance.index);
plt.tight_layout()
