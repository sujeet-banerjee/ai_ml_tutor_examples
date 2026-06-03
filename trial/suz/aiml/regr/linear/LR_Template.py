import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Prepare the Data
# Scikit-learn requires the features (x) to be a 2D array (a column vector).
# We use .reshape(-1, 1) to convert the 1D list into a 2D column.
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(10, 1)
y = np.array([2.8, 5.1, 7.0, 9.2, 11.1, 12.9, 15.2, 17.0, 18.9, 21.1])

print("X=", x)
print("y=", y)

# 2. Initialize and Train the Model
# Since you requested no train/test split, we fit the model on the entire dataset.
model = LinearRegression()
model.fit(x, y)

# 3. Make Predictions
# We use the trained model to predict 'y' values based on our 'x' inputs
y_pred = model.predict(x)

# 4. Extract Metrics
# Get the slope (m), intercept (c), R-squared, and RMSE
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(x, y)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Print the results
print("--- Linear Regression Results ---")
print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"RMSE:      {rmse:.4f}")

# 5. Visualize the Data and the Line of Best Fit
plt.figure(figsize=(8, 5))

# Plot the original data points
plt.scatter(x, y, color='blue', label='Actual Data Points')

# Plot the regression line
plt.plot(x, y_pred, color='red', linewidth=2, label=f'Regression Line (y={slope:.2f}x + {intercept:.2f})')

# Add labels and styling
plt.title('Linear Regression Fit')
plt.xlabel('x (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()