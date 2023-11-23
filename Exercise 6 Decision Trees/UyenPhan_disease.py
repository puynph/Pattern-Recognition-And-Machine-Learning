
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load training and test data
X_test = np.loadtxt('disease_X_test.txt')
y_test = np.loadtxt('disease_y_test.txt')

X_train = np.loadtxt('disease_X_train.txt')
y_train = np.loadtxt('disease_y_train.txt')

# (a) Baseline
baseline_prediction = np.mean(y_train)
baseline_predictions = np.full(y_test.shape, baseline_prediction)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
print("Baseline MSE:", baseline_mse)

# (b) Linear model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_predictions)
print("Linear Model MSE:", linear_mse)

# (c) Decision tree regressor
tree_regressor = DecisionTreeRegressor()
tree_regressor.fit(X_train, y_train)
tree_predictions = tree_regressor.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_predictions)
print("Decision Tree Regressor MSE:", tree_mse)

# (d) Random forest regressor
random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(X_train, y_train)
rf_predictions = random_forest_regressor.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print("Random Forest Regressor MSE:", rf_mse)

