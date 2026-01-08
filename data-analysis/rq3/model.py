import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder



def predict_performance(steps:int, data_path:str):
    pass

data_path = "C:/Users\levay\PycharmProjects\project2-1_group15\data-collection\data"

static_df = pd.read_csv(data_path+"/static.csv")
summary_df = pd.read_csv(data_path+"/summary.csv")


df = pd.merge(static_df, summary_df[['RunID', 'final_reward']], on='RunID')

initial_count = len(df)
df = df.dropna(subset=['final_reward'])
print(f"Dropped {initial_count - len(df)} rows due to NaN in final_reward.")


hw_features = ['os', 'cpu', 'physical_cores', 'system_ram_gb', 'gpu', 'GPU_ram_mb']

hp_features = [
    'batch_size', 'learning_rate', 'buffer_size', 'beta', 'epsilon', 'lambd',
    'num_epoch', 'hidden_units', 'num_layers', 'reward_extrinsic_gamma',
    'reward_extrinsic_strength', 'time_horizon'
]

features = hw_features + hp_features
target = 'final_reward'


df = df[df['max_steps'] == 000000]


X = df[features].copy()
y = df[target]

# Convert categorical strings (CPU, GPU, OS) to numeric labels
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))

# Fill missing values (e.g., if some hyperparameters were not used)
X = X.fillna(0)

# 5. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
mean_reward = y.mean()
mean_reward_prediction = predictions.mean()
mae = mean_absolute_error(y_test, predictions)

print(f"Results for 1M Training Steps:")
print(f"Mean Reward: {mean_reward:.2f}")
print(f"Mean Reward Prediction: {mean_reward_prediction:.2f}")
print(f"MAE: {mae:.2f}%")


