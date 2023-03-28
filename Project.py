"""# **Load Data**"""

import pandas as pd

df=pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv") ## Reads the enclosed CSV file
df

"""#**Data Visualization**"""

df.describe() #Shows the data distribution, mean, count, std, min and max

df.info() #shows data info

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""##Pandas Profiling

Pandas profiling is a library that generates interactive reports with out that, we can see the distribution of the data, the data types and possible problems it might have. It can be sent to anyone.
"""

# ! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
prof = ProfileReport(df)
prof.to_file(output_file='report.html')

"""Depois da cell acima ter corrido, ele cria um file chamado "report.html" do qual pode ser feito o download e, depois, analisado para ver counts, missings e tudo mais.

##MatPlotLib
"""

import matplotlib.pyplot as plt
plt.plot(df)

"""# **Data preparation**

### Data separation as X and Y
"""

y = df['logS'] # Defines 'logS' as the Y (independent/output/target) variable
y

X = df.drop('logS', axis=1) # O "axis=1" permite-nos trabalhar com os dados como colunas, se fosse "axis=0" trabalhava por rows.
X

"""### Data splitting"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

X_train

X_test

"""# **Model building**

## **Linear Regression**

## **Training the model**
"""

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

"""## **Applying the model to make a prediction** """

y_lr_train_pred = lr.predict(X_train) 
y_lr_test_pred = lr.predict(X_test)

y_lr_train_pred

y_lr_test_pred

"""## **Evaluate model performance**"""

from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)

lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

lr_results
