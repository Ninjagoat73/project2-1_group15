import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import CsvMerger
import sys
import yaml

def read_config_as_dict(config_file_path : str):
    conf_dict = {
        "random_state": 42,
        "n_estimators": 100,
        "max_depth":15,
        "min_samples_leaf":6,
        "n_jobs" : -1,
        "random_state_model" : 42,
        "cv_folds" : 10
        }
    with open(config_file_path, 'r', encoding='utf-8-sig') as f:
        try:
            full_yaml = yaml.safe_load(f)["rq4"]
            print(full_yaml)
            conf_dict["random_state"] = full_yaml["random_state"]
            conf_dict["n_estimators"] = full_yaml["n_estimators"]
            conf_dict["max_depth"] = full_yaml["max_depth"]
            conf_dict["min_samples_leaf"] = full_yaml["min_samples_leaf"]
            conf_dict["max_features"] = full_yaml["max_features"]
            conf_dict["n_jobs"] = full_yaml["n_jobs"]
            conf_dict["random_state_model"] = full_yaml["random_state_model"]
            conf_dict["cv_folds"] = full_yaml["cv_folds"]
            return conf_dict
        except yaml.YAMLError as exc:
            print(exc)
            return conf_dict

if len(sys.argv) < 2:
    print("Usage: python rq4.py <path_to_config_file>")
    sys.exit(0)

config_file_path = sys.argv[1]
conf_dict = read_config_as_dict(config_file_path)
global RANDOM_STATE
RANDOM_STATE = conf_dict["random_state"]
print(RANDOM_STATE)
    
CsvMerger.generate_input()

df = pd.read_csv('input.csv')
df = df.sample(frac=1, random_state=conf_dict["random_state"]).reset_index(drop=True)

# Feature setup
numeric_features = ['final_reward','mean_cpu_frequency', 'physical_cores','system_ram_gb', 'GPU_ram_mb','cpu_usage_percent',
                    'game_cpu_usage','max_cpu_frequency','max_game_cpu_usage_percent','mean_game_memory_usage_mb',
                    'mean_game_cpu_usage_percent','max_game_memory_usage_mb','game_memory_usage(MB)','ram_usage_mb',
                    'max_cpu_usage_percent','mean_ram_used_mb','max_ram_used_mb','mean_cpu_usage_percent']
text_features = ['game_name', 'os', 'cpu', 'gpu']
feature_columns = text_features + numeric_features

# Target setup
df['normalize'] = df['normalize'].fillna(0)
df['normalize'] = df['normalize'].astype(int)
text_target_columns = ['training_type', 'learning_rate_schedule', 'vis_encode_type']
target_encoders = {}

for col in text_target_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    target_encoders[col] = le

numeric_target_columns = ['batch_size', 'learning_rate_y', 'beta_y', 'epsilon_y', 'lambd', 'num_epoch', 'hidden_units', 'num_layers', 'keep_checkpoints', 'time_horizon' ,'summary_freq']

target_columns = numeric_target_columns + text_target_columns + ['normalize']
df[target_columns] = df[target_columns].fillna(0)


X = df[feature_columns]
y = df[target_columns]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=conf_dict["random_state"])

preprocessor = ColumnTransformer(transformers=[
    ('text_idx', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), text_features),
    ('num_scale', StandardScaler(), numeric_features)
])

rf_regularized = RandomForestRegressor(
    n_estimators = conf_dict["n_estimators"],
    max_depth = conf_dict["max_depth"],
    min_samples_leaf = conf_dict["min_samples_leaf"],
    max_features = conf_dict["max_features"],
    n_jobs=conf_dict["n_jobs"],
    random_state=conf_dict["random_state_model"]
)

pipeline = Pipeline(steps=[
    ('prep', preprocessor),
    ('regressor', TransformedTargetRegressor(
        regressor = MultiOutputRegressor(rf_regularized),
        transformer = StandardScaler()
    ))
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv = conf_dict["cv_folds"], scoring ='r2')
cv_scores2 = -cross_val_score(pipeline, X_train, y_train, cv = conf_dict["cv_folds"], scoring ='neg_mean_squared_error')
avg_cv_r2 = np.mean(cv_scores)
avg_cv_MSE = np.mean(cv_scores2)
print(f"Overall CV R2 Score: {avg_cv_r2:.4f}")
print(f"Overall CV MSE  Score: {avg_cv_MSE:.4f}")

# TRAIN THE GLOBAL MODEL
print("Training Master Model on all games...")
pipeline.fit(X_train, y_train)

# Predict both sets to check for overfitting
y_pred_train = pipeline.predict(X_train)
y_pred_val = pipeline.predict(X_val)

unique_games = X_val['game_name'].unique()
summary_report = []

print(f"\n{'Game':<20} | {'Train R2':<10} | {'Val R2':<10} | {'Val RMSE':<10} | {'Rows':<6}")
print("-" * 75)

for game in unique_games:

    # Filter the validation set for this specific game

    mask_train = X_train['game_name'] == game
    mask_val = X_val['game_name'] == game

    # Check if game exists in both splits
    if mask_train.sum() < 2 or mask_val.sum() < 2: continue

    # Metrics
    r2_train = r2_score(y_train[mask_train], y_pred_train[mask_train])
    r2_val = r2_score(y_val[mask_val], y_pred_val[mask_val])
    rmse_val = np.sqrt(mean_squared_error(y_val[mask_val], y_pred_val[mask_val]))

    print(f"{game:<20} | {r2_train:>10.4f} | {r2_val:>10.4f} | {rmse_val:>10.4f} | {mask_val.sum():>6}")

    summary_report.append({
        'Game': game,
        'Train_R2': r2_train,
        'Val_R2': r2_val,
        'RMSE': rmse_val,
        'Rows': mask_val.sum()
    })


# PER-FEATURE RMSE
print("\n" + "="*30)
print("PER-FEATURE ERROR (ALL GAMES)")
print("="*30)
for i, col_name in enumerate(target_columns):
    feature_rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], y_pred_val[:, i]))
    print(f"{col_name:<25} | RMSE: {feature_rmse:.6f}")

# PLOTTING PERFORMANCE
summary_df = pd.DataFrame(summary_report).sort_values(by='Val_R2', ascending=False)

x = np.arange(len(summary_df['Game']))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - width/2, summary_df['Train_R2'], width, label='Train R2', color='skyblue', edgecolor='black')
ax.bar(x + width/2, summary_df['Val_R2'], width, label='Val R2', color='salmon', edgecolor='black')

ax.set_ylabel('R2 Score')
ax.set_title(f'Final Model: Train vs Val Performance (Global CV: {avg_cv_r2:.3f})')
ax.set_xticks(x)
ax.set_xticklabels(summary_df['Game'], rotation=45, ha='right')
ax.legend()
ax.axhline(0, color='black', linewidth=0.8)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



