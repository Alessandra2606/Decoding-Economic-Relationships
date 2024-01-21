# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:54:14 2024

@author: utente411
"""

!pip install linearmodels

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS
import seaborn as sns
sns.set_theme()

conda install -c conda-forge xlrd>=2.0.1

df1 = pd.read_excel("C:/Users/utente411/Desktop/database.xls", index_col=0)
df1.head()          

df1.plot(x='GDP', y='Price deflator', kind='scatter')
plt.show()

df1_subset = df1.dropna(subset=['Price deflator', 'GDP'])


X = df1_subset['GDP']
y = df1_subset['Price deflator']
labels = df1_subset['Graph']

# Replace markers with country labels
fig, ax = plt.subplots()
ax.scatter(X, y, marker='')

for i, label in enumerate(labels):
    ax.annotate(label, (X.iloc[i], y.iloc[i]))

# Fit a linear trend line
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
         color='black')

ax.set_xlim([-0,9]) 
ax.set_ylim([-0,6])
ax.set_xlabel('GDP')
ax.set_ylabel('Price deflator')
ax.set_title('Figure 2: OLS relationship between Price deflator \
and GDP')
plt.show()

df1['const'] = 1

reg1 = sm.OLS(endog=df1['Price deflator'], exog=df1[['const', 'GDP']], \
    missing='drop')
type(reg1)

results = reg1.fit()
type(results)

print(results.summary())

mean_expr = np.mean(df1_subset['GDP'])
mean_expr

predicted_GDP = 0.3219 + 0.4707 * 5.9222
predicted_GDP

results.predict(exog=[1, mean_expr])

df1_plot = df1.dropna(subset=['Price deflator', 'GDP'])

# Plot predicted values

fix, ax = plt.subplots()
ax.scatter(df1_plot['GDP'], results.predict(), alpha=0.5,
        label='predicted')

# Plot observed values

ax.scatter(df1_plot['GDP'], df1_plot['Price deflator'], alpha=0.5,
        label='observed')

ax.legend()
ax.set_title('OLS predicted values')
ax.set_xlabel('GDP')
ax.set_ylabel('Price deflator')
plt.show()

df2 = pd.read_excel("C:/Users/utente411/Desktop/database.xls", index_col=0)

# Add constant term to dataset
df2['const'] = 1

# Create lists of variables to be used in each regression
X1 = ['const', 'GDP']
X2 = ['const', 'GDP', 'Employment Rate']
X3 = ['const', 'GDP', 'Employment Rate', 'Labor Productivity']
X4 = ['const', 'GDP', 'Employment Rate', 'Labor Productivity', 'Debt Ratio']

# Estimate an OLS regression for each set of variables
reg1 = sm.OLS(df2['Price deflator'], df2[X1], missing='drop').fit()
reg2 = sm.OLS(df2['Price deflator'], df2[X2], missing='drop').fit()
reg3 = sm.OLS(df2['Price deflator'], df2[X3], missing='drop').fit()
reg4 = sm.OLS(df2['Price deflator'], df2[X4], missing='drop').fit()

info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[reg1,reg2,reg3,reg4],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 1',
                                         'Model 2',
                                         'Model 3',
                                         'Model 4'],
                            info_dict=info_dict,
                            regressor_order=['const',
                                             'GDP',
                                             'Employment Rate',
                                             'Labor Productivity',
                                             'Debt Ratio'])

results_table.add_title('Table 2 - OLS Multiple Regressions')

print(results_table)

import seaborn as sns
import matplotlib.pyplot as plt


numeric_df1 = df1.select_dtypes(include='number')
correlation_matrix = numeric_df1.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Matrice di Correlazione")
plt.show()


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


X = sm.add_constant(df1[['Price deflator', 'GDP', 'Employment Rate', 'Labor Productivity', 'Debt Ratio']])

model = sm.OLS(df1['Price deflator'], X).fit()

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)



