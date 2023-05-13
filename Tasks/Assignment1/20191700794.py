import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# Loading data
df = pd.read_csv('SuperMarketSales.csv')

def date_to_float(dt):
    # Calculating the months and days
    calc = (((dt.month - 1) * 30) + dt.day) / 365
    # Adding calc to the years
    return dt.year + calc

# Parsing the date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Date'] = df['Date'].apply(date_to_float)   
            
# Eliminate weekly sales column outliers
df = df[(np.abs(stats.zscore(df['Weekly_Sales'])) < 3)]

Y = df['Weekly_Sales']
df = df.drop(['Weekly_Sales'], axis=1)

df_results = pd.DataFrame(columns=['Column', 'Mean Square Error', 'R2 Score'])

for column in df.columns:
    print(column)
    X = df[column].values.reshape(-1,1)
    
    cls = linear_model.LinearRegression()
    cls.fit(X,Y)
    prediction = cls.predict(X)
    
    # Plotting
    plt.scatter(X, Y)
    plt.xlabel(column, fontsize = 20)
    plt.ylabel('Weekly Sales', fontsize = 20)
    plt.plot(X, prediction, color='red', linewidth = 3)
    # plt.show()
    
    # Printing the metrics
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
    print('R2 Score', metrics.r2_score(Y, prediction))
    
    df_results = df_results.append({'Column': column, 'Mean Square Error': metrics.mean_squared_error(Y, prediction), 'R2 Score': metrics.r2_score(Y, prediction)}, ignore_index=True)
    
    print('------------------------------------')



print(df_results.sort_values(by=['Mean Square Error'], ascending=True))