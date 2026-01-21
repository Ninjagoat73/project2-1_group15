import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def scale_group(group):
    s = StandardScaler()
    return s.fit_transform(group.values.reshape(-1, 1)).flatten()

def global_model(config_path):

    config_dict = read_config_as_dict(config_path)

    data_path = config_dict['data_path']

    static_df = pd.read_csv(data_path + '/static.csv')
    summary_df = pd.read_csv(data_path + '/summary.csv')

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

    merged_df['scaled_reward'] = merged_df.groupby('game_name')['final_reward'].transform(scale_group)

    X_global = merged_df[feature_cols].copy()
    y_global = merged_df['scaled_reward'].fillna(0)

    le_game = LabelEncoder()
    X_global['game_name'] = le_game.fit_transform(X_global['game_name'])

    le_sched = LabelEncoder()
    X_global['learning_rate_schedule'] = le_sched.fit_transform(X_global['learning_rate_schedule'])

    X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=config_dict["test_size"], random_state= config_dict["random_state"] , stratify=merged_df['game_name'])
    regr = RandomForestRegressor(n_estimators=config_dict["n_estimators"], max_depth=config_dict["max_depth"], min_samples_leaf=config_dict["min_samples_leaf"], random_state=config_dict["random_state"])
    regr.fit(X_train, y_train)
    y_train_pred = regr.predict(X_train)
    global_train_r2 = r2_score(y_train, y_train_pred)
    print(f"Global Training R2: {global_train_r2:.2f}")
    game_counts = static_df["game_name"].value_counts()

    for game in games:

        game_id = le_game.transform([game])[0]

        game_mask = X_test['game_name'] == game_id

        if not game_mask.any():
            print(f"-----------{game}------------")
            print("Not enough samples in test set to calculate R2.")
            r2_dict[game] = 0
            r2_train_dict[game] = 0
            continue

        game_X_test = X_test[game_mask]
        game_y_test = y_test[game_mask]

        predictions = regr.predict(game_X_test)
        r2 = r2_score(game_y_test, predictions)
        mae = mean_absolute_error(game_y_test, predictions)

        r2_dict[game] = r2

        print(f"-----------{game}------------")
        print(f"Prediction Accuracy (R2): {r2:.2f}")
        print(f"Average Error (MAE): {mae:.2f}")
        print(f"Game Counts: {game_counts[game]:.2f}")

        r2_dict.update({game: r2})
        mae_dict.update({game: mae})

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


def read_config_as_dict(config_path : str):
    conf_dict = {
        "random_state": 1,
        "data_path": "../../data-collection/data",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "test_size": 0.20,
    }
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        try:
            full_yaml = yaml.safe_load(f)['random_forest']
            print(full_yaml)
            conf_dict["random_state"] = full_yaml["random_state"]
            conf_dict["data_path"] = full_yaml["data_path"]
            conf_dict["n_estimators"] = full_yaml["n_estimators"]
            conf_dict["max_depth"] = full_yaml["max_depth"]
            conf_dict["min_samples_leaf"] = full_yaml["min_samples_leaf"]
            conf_dict["test_size"] = full_yaml["test_size"]
            return conf_dict
        except Exception as exc:
            print("Error:" + exc)
            print("There was an error reading config file, using default parameters.")
            return conf_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python per_game_model.py <path_to_config_file>")
    else:
        global_model(sys.argv[1])
