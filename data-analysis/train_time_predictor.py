# RQ1: can we predict wall-clock time based on hardware + hyperparameters?
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("training_data_prepared.csv")
print("loaded", len(data), "samples")

# features: hardware specs  hyperparameters
# hardware: physical_cores ram_gb
# hyperparams: batch_size learning_rate....
features = ['physical_cores', 'ram_gb',
            'batch_size', 'learning_rate', 'max_steps', 'buffer_size',
            'hidden_units', 'num_layers', 'num_epoch', 'time_horizon']

target = 'total_time_seconds'
X = data[features]
y = data[target]

# drop missing values
X = X.dropna()
y = y[X.index]

print("using", len(X), "samples after cleaning")

# split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("train:", len(X_train), "test:", len(X_test))

# try linear regression first
print("\nLinear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print("MAE:", round(lr_mae, 2), "seconds")
print("R2:", round(lr_r2, 3))

# now try random forest
print("\nRandom Forest")
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("MAE:", round(rf_mae, 2), "seconds")
print("R2:", round(rf_r2, 3))

# plot results for the best model
if rf_r2 > lr_r2:
    best_pred = rf_pred
    best_name = "Random Forest"
    best_mae = rf_mae
    best_r2 = rf_r2
else:
    best_pred = lr_pred
    best_name = "Linear Regression"
    best_mae = lr_mae
    best_r2 = lr_r2

plt.figure(figsize=(8,6))
plt.scatter(y_test, best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Time (s)')
plt.ylabel('Predicted Time (s)')
plt.title(f'{best_name}\nMAE: {best_mae:.1f}s, R2: {best_r2:.3f}')
plt.tight_layout()
plt.savefig('time_prediction_results.png')
print(f"\nPlot saved to time_prediction_results.png")
