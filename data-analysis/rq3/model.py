import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sympy import series

data_path = "C:/Users\levay\PycharmProjects\project2-1_group15\data-collection\data/"

static_df = pd.read_csv(data_path + 'static.csv')
summary_df = pd.read_csv(data_path + 'summary.csv')

merged_df = pd.merge(static_df, summary_df[['RunID', 'final_reward']], on='RunID')

games = merged_df['game_name'].unique()

r2_dict = {}
mae_dict = {}

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

    print(f"-----------{game}------------")
    print(f"Prediction Accuracy (R2): {r2:.2f}")
    print(f"Average Error (MAE): {mae:.2f}")

    if game == "Pyramids":
        r2_dict.update({game: 0})
        mae_dict.update({game: mae})
    else:
        r2_dict.update({game: r2})
        mae_dict.update({game: mae})


    #importances = pd.Series(regr.feature_importances_, index=feature_cols).sort_values(ascending=False)
    #print("\nMost Important Hyperparameters:")
    #print(importances.head(5))


game_counts = static_df["game_name"].value_counts()
r2_scores = list(r2_dict.values())

plt.plot(games, r2_scores, game_counts)

plt.title("R2 vs. MAE")
plt.ylabel("R2")
plt.xlabel("Game")

plt.show()

