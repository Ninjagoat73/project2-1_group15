# predict training time using linear regression
# RQ1: can we predict wall-clock time based on hardware + hyperparameters?

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = pd.read_csv("training_data_prepared.csv")
print("loaded", len(data), "samples")
# pick features to use for prediction
features = ['batch_size', 'learning_rate', 'max_steps', 'buffer_size',
            'hidden_units', 'num_layers', 'num_epoch', 'time_horizon']

# target is the total time
target = 'total_time_seconds'
X = data[features]
y = data[target]

# fallback for now just dropping the missing values
X = X.dropna()
y = y[X.index]

print("using", len(X), "samples after removing missing")

# splitting data 80% is trainng adn 20% is testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("train:", len(X_train), "test:", len(X_test))

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# predict on test
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nResults:")
print("MAE:", round(mae, 2), "seconds")
print("R2:", round(r2, 3))

