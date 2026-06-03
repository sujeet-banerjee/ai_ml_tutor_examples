import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train = pd.get_dummies(pd.read_csv('regression_discrete.csv'))
val = pd.get_dummies(pd.read_csv('regression_discrete_val.csv'))
train, val = train.align(val, join='left', axis=1, fill_value=0)

X_train, y_train = train.drop('Monthly_Support_Tickets', axis=1), train['Monthly_Support_Tickets']
X_val, y_val = val.drop('Monthly_Support_Tickets', axis=1), val['Monthly_Support_Tickets']

model = RandomForestRegressor().fit(X_train, y_train)
mae = mean_absolute_error(y_val, model.predict(X_val))
print(f"Discrete Regression MAE: {mae:.2f} tickets")