import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sympy import series

data_path = "E:\project2-1\data-collection\data/"

static_df = pd.read_csv(data_path + 'static.csv')
summary_df = pd.read_csv(data_path + 'summary.csv')

merged_df = pd.merge(static_df, summary_df[['RunID', 'final_reward']], on='RunID')

games = merged_df['game_name'].unique()

r2_dict = {}
mae_dict = {}

optimal_settings = []

for game in games:

    game_df = merged_df[merged_df['game_name'] == game].copy()

    feature_cols = [
    'batch_size', 'learning_rate', 'buffer_size', 'beta',
    'epsilon', 'lambd', 'num_epoch', 'learning_rate_schedule',
    'normalize', 'hidden_units', 'num_layers', 'max_steps'
    ]

    X = game_df[feature_cols].copy()
    y = game_df['final_reward']
    y = y.fillna(0)

    le = LabelEncoder()
    X['learning_rate_schedule'] = le.fit_transform(X['learning_rate_schedule'])
    X['normalize'] = X['normalize'].astype(int)

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    regr = RandomForestRegressor(n_estimators=100)
    regr.fit(X_train, y_train)

    predictions = regr.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    print(f"-----------{game}------------")
    print(f"Prediction Accuracy (R2): {r2:.2f}")
    print(f"Average Error (MAE): {mae:.2f}")
    print(f"Average Error Percentage (MAPE): {mape:.2f}%")

    if game == "Pyramids":
        r2_dict.update({game: 0})
        mae_dict.update({game: mae})
    else:
        r2_dict.update({game: r2})
        mae_dict.update({game: mae})

    if r2 > 0.7:
        samples = 100000
        search_space = {}

        for col in feature_cols:
            if col == 'learning_rate_schedule':
                search_space[col] = np.random.choice(X[col].unique(), samples)
            elif col == 'max_steps':
                search_space[col] = np.random.choice([100000, 250000, 500000, 1000000, 2000000], samples)
            elif col == 'normalize':
                search_space[col] = np.random.choice([0, 1], samples)
            elif col in ['num_layers', 'hidden_units', 'num_epoch']:
                search_space[col] = np.random.randint(X[col].min(), X[col].max() + 1, samples)
            else:
                search_space[col] = np.random.uniform(X[col].min(), X[col].max(), samples)

        made_X =pd.DataFrame(search_space)
        made_predictions = regr.predict(made_X)
        best_idx = np.argmax(made_predictions)
        best_hyperparameters = made_X.iloc[best_idx].to_dict()
        best_reward = made_predictions[best_idx]

        best_hyperparameters['learning_rate_schedule'] = le.inverse_transform([int(best_hyperparameters['learning_rate_schedule'])])[0]
        best_hyperparameters['predicted_reward'] = best_reward
        best_hyperparameters['game_name'] = game
        best_hyperparameters['model_r2'] = r2

        optimal_settings.append(best_hyperparameters)


    #importances = pd.Series(regr.feature_importances_, index=feature_cols).sort_values(ascending=False)
    #print("\nMost Important Hyperparameters:")
    #print(importances.head(5))

if optimal_settings:
    optimal_df = pd.DataFrame(optimal_settings)

    cols = ['game_name', 'predicted_reward', 'model_r2'] + feature_cols
    optimal_df = optimal_df[cols]


    print("-----------Optimal Hyperparams-+----------")

    print(optimal_df.to_string(index=False))


game_counts = static_df["game_name"].value_counts()

plot_data = pd.DataFrame({
    'Game': games,
    'R2': [r2_dict[g] for g in games],
    'Count': [game_counts[g] for g in games]
}).sort_values('Count', ascending=True)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.bar(plot_data['Game'], plot_data['Count'], color='skyblue', alpha=0.6, label='Sample Count')
ax1.set_xlabel('Game Name')
ax1.set_ylabel('Number of Occurrences (Counts)', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
plt.xticks(rotation=45, ha='right')


ax2 = ax1.twinx()
ax2.plot(plot_data['Game'], plot_data['R2'], color='red', marker='o', linewidth=2, label='R2 Score')
ax2.set_ylabel('Prediction Accuracy ($R^2$)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(-0.5, 1.1)

plt.title('Dataset Size vs. Model Accuracy ($R^2$) per Game', fontsize=14)
fig.tight_layout()

plt.show()
