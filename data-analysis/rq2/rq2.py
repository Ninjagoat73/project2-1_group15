# all the imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import yaml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from sklearn.linear_model import LinearRegression, MultiTaskLasso, QuantileRegressor, RANSACRegressor, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

if len(sys.argv) < 2:
    print("Usage: python clustering.py <path_to_config_file>")
    sys.exit(0)
    
def cross_validate_models(model_list, preprocessor, x, y):
    scoring = {
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for model_name, regressor in model_list.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", regressor)
            ]
        )
        scores = cross_validate(pipeline,X,y,cv=cv,scoring=scoring,n_jobs=-1)
        results.append({
            "Model": model_name,
            "MAE (MB)": -scores["test_mae"].mean(),
            "R²": scores["test_r2"].mean()
        })
    return results

def train_models(model_list, preprocessor, X_train, y_train, X_test, y_test):
    results = []
    i = 0
    for model_name, regressor in model_list.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", regressor)
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results.append({
            "Model": model_name,
            "MAE (MB)": mean_absolute_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred)
        })
    return results

def read_config_as_dict(config_file_path : str):
    conf_dict = {
        "random_state": 42,
        "randomforest": {
            "n_estimators": 300,
            "forest_random_state": 42,
            "n_jobs": -1
        },
        "gradientboosting": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "boosting_random_state": 42,
        }
    }
    with open(config_file_path, 'r', encoding='utf-8-sig') as f:
        try:
            full_yaml = yaml.safe_load(f)["rq2"]
            print(full_yaml)
            conf_dict["random_state"] = full_yaml["random_state"]
            conf_dict["randomforest"]["n_estimators"] = full_yaml["randomforest"]["n_estimators"]
            conf_dict["randomforest"]["forest_random_state"] = full_yaml["randomforest"]["forest_random_state"]
            conf_dict["randomforest"]["n_jobs"] = full_yaml["randomforest"]["n_jobs"]
            conf_dict["gradientboosting"]["max_depth"] = full_yaml["gradientboosting"]["max_depth"]
            conf_dict["gradientboosting"]["learning_rate"] = full_yaml["gradientboosting"]["learning_rate"]
            conf_dict["gradientboosting"]["boosting_random_state"] = full_yaml["gradientboosting"]["boosting_random_state"]
            return conf_dict
        except yaml.YAMLError as exc:
            print(exc)
            return conf_dict

           

static_df = pd.read_csv("../../data-collection/data/static.csv")
summary_df = pd.read_csv("../../data-collection/data/summary.csv")

static_df["system_ram_mb"]= static_df["system_ram_gb"] / (1024 * 1024)
static_df = static_df.drop(columns=["system_ram_gb"])
static_df["GPU_ram_mb"] = static_df["GPU_ram_mb"].fillna(
    static_df["system_ram_mb"]
)
all_nan_cols = static_df.columns[static_df.isna().all()]
print("Dropping all-NaN columns:", list(all_nan_cols))
static_df = static_df.drop(columns=all_nan_cols)
try:
    static_df = static_df.drop(columns=['reward_extrinsic_gamma', 'reward_extrinsic_strength'])
except:
    pass
static_df = static_df.dropna()
summary_df = summary_df.dropna()
df = static_df.merge(
    summary_df[["RunID", "mean_ram_used_mb", "max_ram_used_mb"]],
    on="RunID",
    how="inner"
)
print(df.shape)
print(df.head)
targets = [
    "mean_ram_used_mb",
    "max_ram_used_mb"
]
drop_columns = ["summary_freq", "RunID"]
X = df[static_df.columns.drop(drop_columns)]
y_mean = df["mean_ram_used_mb"]
y_max = df["max_ram_used_mb"]

categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "bool"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

config_file_path = sys.argv[1]
config_dict = read_config_as_dict(config_file_path)
global RANDOM_STATE
RANDOM_STATE = config_dict["random_state"]
print(RANDOM_STATE)
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=config_dict["randomforest"]["n_estimators"],
        random_state=config_dict["randomforest"]["forest_random_state"],
        n_jobs=config_dict["randomforest"]["n_jobs"]
    ),
    "GradientBoosting": HistGradientBoostingRegressor(
        max_depth=config_dict["gradientboosting"]["max_depth"],
        learning_rate=config_dict["gradientboosting"]["learning_rate"],
        random_state=config_dict["gradientboosting"]["boosting_random_state"]
    ),
    "RANSACRegressor": RANSACRegressor(),
}

X_train, X_test, y_train, y_test = train_test_split(X, y_mean, test_size=0.2, random_state=RANDOM_STATE)
results = train_models(models, preprocessor, X_train, y_train, X_test, y_test)
print(results)
results = cross_validate_models(models, preprocessor, X, y_mean)
print(results)

X_train, X_test, y_train, y_test = train_test_split(X, y_max, test_size=0.2, random_state=RANDOM_STATE)
results = cross_validate_models(models, preprocessor, X, y_max)
print(results)
results = train_models(models, preprocessor, X_train, y_train, X_test, y_test)
print(results)