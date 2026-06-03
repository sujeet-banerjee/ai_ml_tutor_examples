import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from trial.suz.aiml.regr.data import \
    data_csv_house_price, data_csv_house_price_val

# Load and Preprocess
## get_dummies ==> convert categorical to dummy variables
train = pd.get_dummies(data_csv_house_price)
val = pd.get_dummies(data_csv_house_price_val)

# Align columns (ensure same one-hot features)
train, val = train.align(val, join='left', axis=1, fill_value=0)

# Split features and target
X_train, y_train = train.drop('Price_USD', axis=1), train['Price_USD']
X_val, y_val = val.drop('Price_USD', axis=1), val['Price_USD']

# Train and Score
model = LinearRegression().fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
print(f"Regression RMSE: ${rmse:,.2f}")