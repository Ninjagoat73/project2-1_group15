import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('data-analysis/input.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# text,float, float, text, text, int, float, text, float
numeric_features = ['final_reward','cpu_frequency_mhz', 'physical_cores','system_ram_gb', 'GPU_ram_mb']
text_features = ['game_name', 'os', 'cpu', 'gpu']

feature_columns = text_features + numeric_features

df['normalize'] = df['normalize'].astype(int)

text_target_columns = ['training_type', 'learning_rate_schedule', 'vis_encode_type']
target_encoders = {}


for col in text_target_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    target_encoders[col] = le

joblib.dump(target_encoders, 'target_encoders.joblib')

numeric_target_columns = ['batch_size', 'learning_rate_y', 'buffer_size', 'beta_y', 'epsilon_y', 'lambd', 'num_epoch', 'hidden_units', 'num_layers', 'keep_checkpoints', 'max_steps', 'time_horizon' ,'summary_freq']

target_columns = numeric_target_columns + text_target_columns + ['normalize']

unique_games = df['game_name'].unique()
summary_report = []

df[target_columns] = df[target_columns].fillna(0)

for game in unique_games:
    game_data = df[df['game_name'] == game].copy()
    if len(game_data) < 20 : continue
    
    X = game_data[feature_columns]
    y = game_data[target_columns]


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('text_idx', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), text_features),
        ('num_scale', StandardScaler(), numeric_features)
    ])

    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('regressor', MultiOutputRegressor(
            RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        ))
    ])


    print(f"\n--- Processing Game: {game} ---")
    
    # A. Cross-Validation (on Training set only to prevent data leakage)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    avg_cv_r2 = np.mean(cv_scores)

    # B. Validation Accuracy (Testing on the 20% holdout)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    val_r2 = r2_score(y_val, y_pred)

    print(f"  CV Avg R2: {avg_cv_r2:.4f}")
    print(f"  Holdout Validation R2: {val_r2:.4f}")

    # C. Final Fit & Save (Using ALL data for the final saved model)
    pipeline.fit(X, y)
    clean_name = str(game).replace(" ", "_").replace("/", "_")
    joblib.dump(pipeline, f"model_{clean_name}.joblib")
    
    summary_report.append({
        'Game': game, 
        'CV_Avg_R2': avg_cv_r2, 
        'Val_R2': val_r2, 
        'Rows': len(game_data)
    })

    # 5. FINAL REPORT
    print("\n" + "="*60)
    print("                 FINAL PERFORMANCE SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(summary_report)
    print(summary_df.to_string(index=False))