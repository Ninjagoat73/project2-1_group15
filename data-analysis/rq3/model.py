import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sympy import series


def scale_group(group):
    s = StandardScaler()
    return s.fit_transform(group.values.reshape(-1, 1)).flatten()

data_path = "E:\project2-1\data-collection\data/"

static_df = pd.read_csv(data_path + 'static.csv')
summary_df = pd.read_csv(data_path + 'summary.csv')

merged_df = pd.merge(static_df, summary_df[['RunID', 'final_reward']], on='RunID')

games = merged_df['game_name'].unique()

r2_dict = {}
mae_dict = {}
r2_train_dict = {}
feature_cols = [
    'batch_size', 'learning_rate', 'buffer_size', 'beta',
    'epsilon', 'lambd', 'num_epoch', 'learning_rate_schedule',
    'normalize', 'hidden_units', 'num_layers', 'max_steps', 'game_name',
]

optimal_settings = []

merged_df['scaled_reward'] = merged_df.groupby('game_name')['final_reward'].transform(scale_group)

X_global = merged_df[feature_cols].copy()
y_global = merged_df['scaled_reward'].fillna(0)



le_game = LabelEncoder()
X_global['game_name'] = le_game.fit_transform(X_global['game_name'])

le_sched = LabelEncoder()
X_global['learning_rate_schedule'] = le_sched.fit_transform(X_global['learning_rate_schedule'])

X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state= 1 , stratify=merged_df['game_name'])
regr = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=5)
regr.fit(X_train, y_train)
y_train_pred = regr.predict(X_train)
global_train_r2 = r2_score(y_train, y_train_pred)
print(f"Global Training R2: {global_train_r2:.2f}")
game_counts = static_df["game_name"].value_counts()



for game in games:

    # game_df = merged_df[merged_df['game_name'] == game].copy()
    game_id = le_game.transform([game])[0]

    game_mask = X_test['game_name'] == game_id

    if not game_mask.any():
        print(f"-----------{game}------------")
        print("Not enough samples in test set to calculate R2.")
        r2_dict[game] = 0
        r2_train_dict[game] = 0
        continue



    # X = game_df[feature_cols].copy()
    # y = game_df['final_reward']
    # y = y.fillna(0)

    # le = LabelEncoder()
    # X['learning_rate_schedule'] = le.fit_transform(X['learning_rate_schedule'])
    # X['normalize'] = X['normalize'].astype(int)



    # X = X.fillna(0)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    # regr = RandomForestRegressor(n_estimators=100, max_depth=7,
    # min_samples_leaf=5,
    # max_features='sqrt',
    # random_state=1)
    # regr.fit(X_train, y_train)

    game_X_test = X_test[game_mask]
    game_y_test = y_test[game_mask]

    predictions = regr.predict(game_X_test)

    # r2_train = r2_score(y_train, regr.predict(X_train))
    r2 = r2_score(game_y_test, predictions)
    mae = mean_absolute_error(game_y_test, predictions)



    r2_dict[game] = r2
    # r2_train_dict[game] = r2_train

    print(f"-----------{game}------------")
    print(f"Prediction Accuracy (R2): {r2:.2f}")
    # print(f"Prediction Accuracy Training (R2 Training): {r2_train:.2f}")
    print(f"Average Error (MAE): {mae:.2f}")
    print(f"Game Counts: {game_counts[game]:.2f}")


    r2_dict.update({game: r2})
    mae_dict.update({game: mae})

    if r2 > 0.7:
        samples = 100000
        search_space = {}




        for col in feature_cols:
            if col == 'learning_rate_schedule':
                search_space[col] = np.random.choice(X_global[col].unique(), samples)
            elif col == 'max_steps':
                search_space[col] = np.random.choice([100000, 250000, 500000, 1000000, 2000000], samples)
            elif col == 'normalize':
                search_space[col] = np.random.choice([0, 1], samples)
            elif col in ['num_layers', 'hidden_units', 'num_epoch']:
                search_space[col] = np.random.randint(X_global[col].min(), X_global[col].max() + 1, samples)
            else:
                search_space[col] = np.random.uniform(X_global[col].min(), X_global[col].max(), samples)

        made_X =pd.DataFrame(search_space)
        made_X = made_X[feature_cols]
        made_predictions = regr.predict(made_X)
        best_idx = np.argmax(made_predictions)
        best_hyperparameters = made_X.iloc[best_idx].to_dict()




        best_hyperparameters['learning_rate_schedule'] = le_sched.inverse_transform([int(best_hyperparameters['learning_rate_schedule'])])[0]

        best_hyperparameters['game_name'] = game
        best_hyperparameters['model_r2'] = r2

        optimal_settings.append(best_hyperparameters)

    elif mae <  0.4:
        samples = 100000
        search_space = {}





        for col in feature_cols:
            if col == 'learning_rate_schedule':
                search_space[col] = np.random.choice(X_global[col].unique(), samples)
            elif col == 'max_steps':
                search_space[col] = np.random.choice([100000, 250000, 500000, 1000000, 2000000], samples)
            elif col == 'normalize':
                search_space[col] = np.random.choice([0, 1], samples)
            elif col in ['num_layers', 'hidden_units', 'num_epoch']:
                search_space[col] = np.random.randint(X_global[col].min(), X_global[col].max() + 1, samples)
            else:
                search_space[col] = np.random.uniform(X_global[col].min(), X_global[col].max(), samples)

        made_X =pd.DataFrame(search_space)
        made_X = made_X[feature_cols]
        made_predictions = regr.predict(made_X)
        best_idx = np.argmax(made_predictions)
        best_hyperparameters = made_X.iloc[best_idx].to_dict()

        mae_text = f"{mae:.2f}(MAE)"


        best_hyperparameters['learning_rate_schedule'] = le_sched.inverse_transform([int(best_hyperparameters['learning_rate_schedule'])])[0]

        best_hyperparameters['game_name'] = game
        best_hyperparameters['model_r2'] = mae_text

        optimal_settings.append(best_hyperparameters)


    #importances = pd.Series(regr.feature_importances_, index=feature_cols).sort_values(ascending=False)
    #print("\nMost Important Hyperparameters:")
    #print(importances.head(5))

if optimal_settings:
    optimal_df = pd.DataFrame(optimal_settings)

    cols = ['game_name', 'model_r2'] + feature_cols
    optimal_df = optimal_df[cols]


    print("-----------Optimal Hyperparams-+----------")

    print(optimal_df.to_string(index=False))




plot_data = pd.DataFrame({
    'Game': games,
    'R2': [r2_dict[g] for g in games],
    'MAE': [mae_dict.get(g, 0) for g in games],
    'Count': [game_counts[g] for g in games]

}).sort_values('Count', ascending=False)

fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.bar(plot_data['Game'], plot_data['Count'], color='skyblue', alpha=0.6, label='Sample Count')
ax1.set_xlabel('Game Name')
ax1.set_ylabel('Number of Occurrences (Counts)', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
plt.xticks(rotation=45, ha='right')


ax2 = ax1.twinx()
ax2.plot(plot_data['Game'], plot_data['R2'], color='red', marker='o', linewidth=2, label='R2 Score')
ax2.plot(plot_data['Game'], plot_data['MAE'], color='green', marker='s',
         linewidth=2, linestyle='--', label='MAE (Error)')
ax2.set_ylim(-1.1, 1.1)
ax2.set_ylabel('Metric Value (R2 & MAE)', color='black', fontsize=12)
ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.title('Dataset Size vs. Model Accuracy per Game', fontsize=14)
fig.tight_layout()

plt.show()
