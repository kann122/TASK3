"""
Original file is located at
    https://colab.research.google.com/drive/1SxGJNpNIHxgfRp_39P6lS94JzrNUcJih#scrollTo=iAOu2u3eitS2&line=10&uniqifier=1

IMPORTING LIBRARIES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""IMPORTING DATASET"""

sales_data = pd.read_csv('/content/advertising.csv')

sales_data.head()

"""Given dataset consist of the advertising platform and the related sales. Let's visualize each platform"""

sales_data.shape

sales_data.info()

sales_data.isnull().sum()

sales_data.describe()

"""Basic Observation

Avg expense spend is highest on TV

Avg expense spend is lowest on Radio

Max sale is 27 and Min is 1.6
"""

sns.pairplot(sales_data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()

"""Pair Plot Observation

When advertising cost increases in TV Ads the sales will increase as well, While for the newspaper and radio it is bit unpredictable
"""

sales_data['TV'].plot.hist(bins=10,xlabel='TV')

sales_data['Radio'].plot.hist(bins=10,color='green',xlabel='Radio')

sales_data['Newspaper'].plot.hist(bins=10,color='red',xlabel='Newspaper')

"""Histogram Observation"""

sns.heatmap(sales_data.corr(),annot=True)
plt.show()

"""SALES IS HIGHLY COORELATED WITH THE TV

Model Training
"""

X_train,X_test,Y_train,Y_test = train_test_split(sales_data[['TV']],sales_data[['Sales']],test_size=0.3,random_state=0)

print(X_train)

print(Y_train)

print(X_test)

print(Y_test)

model= LinearRegression()
model.fit(X_train,Y_train)

res=model.predict(X_test)
print(res)

model.coef_

model.intercept_

0.05473199* 69.2 + 7.14382225

plt.plot(res)

plt.scatter(X_test,Y_test)
plt.plot(X_test,7.14382225+0.05473199*X_test,'r')
plt.show()

"""Concluding with saying that above mention solution is sucessfully able to predict the sales using advertising dataset"""
