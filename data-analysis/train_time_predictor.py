# RQ1 time prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import sys

# config
config_file = 'config.json'
if len(sys.argv) > 1:
    config_file = sys.argv[1]
f = open(config_file)
cfg = json.load(f)
f.close()

print("config:", config_file)

df = pd.read_csv(cfg['data_file'])
print("loaded", len(df), "samples")

# games to numbers
game_list = df['game_name'].unique()
for g in game_list:
    df['game_' + g] = 0
    df.loc[df['game_name'] == g, 'game_' + g] = 1

# features
feat = []
for f in cfg['features']['hardware']:
    feat.append(f)
for f in cfg['features']['hyperparameters']:
    feat.append(f)
for c in df.columns:
    if c.startswith('game_'):
        feat.append(c)

# get X and y
X = df[feat]
y = df[cfg['target']]

# remove non numeric stuff
cols_to_keep = []
for c in X.columns:
    if X[c].dtype in ['int64', 'float64']:
        cols_to_keep.append(c)
X = X[cols_to_keep]
feat = cols_to_keep

# drop na
X = X.dropna()
y = y[X.index]

print("using", len(X), "samples")
print("features:", len(feat))

# split data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# comparing models
print("\n=== Model Comparison ===")

# linear
lr = LinearRegression()
lr.fit(X_tr, y_tr)
pred_lr = lr.predict(X_te)
r2_lr = r2_score(y_te, pred_lr)
mae_lr = mean_absolute_error(y_te, pred_lr)
r2_lr_train = r2_score(y_tr, lr.predict(X_tr))

# random forest
rf = RandomForestRegressor(
    n_estimators=cfg['models']['random_forest']['n_estimators'],
    max_depth=cfg['models']['random_forest']['max_depth'],
    min_samples_leaf=cfg['models']['random_forest']['min_samples_leaf'],
    random_state=42)
rf.fit(X_tr, y_tr)
pred_rf = rf.predict(X_te)
r2_rf = r2_score(y_te, pred_rf)
mae_rf = mean_absolute_error(y_te, pred_rf)
r2_rf_train = r2_score(y_tr, rf.predict(X_tr))

# gradient boosting
gb = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
gb.fit(X_tr, y_tr)
pred_gb = gb.predict(X_te)
r2_gb = r2_score(y_te, pred_gb)
mae_gb = mean_absolute_error(y_te, pred_gb)
r2_gb_train = r2_score(y_tr, gb.predict(X_tr))

print("Linear: train_r2=" + str(round(r2_lr_train,3)) + " test_r2=" + str(round(r2_lr,3)) + " mae=" + str(round(mae_lr,0)))
print("RF: train_r2=" + str(round(r2_rf_train,3)) + " test_r2=" + str(round(r2_rf,3)) + " mae=" + str(round(mae_rf,0)))
print("GradBoost: train_r2=" + str(round(r2_gb_train,3)) + " test_r2=" + str(round(r2_gb,3)) + " mae=" + str(round(mae_gb,0)))

# feature importance
print("\n=== Feature Importance ===")
importances = rf.feature_importances_
feat_imp = []
for i in range(len(feat)):
    feat_imp.append([feat[i], importances[i]])
feat_imp.sort(key=lambda x: x[1], reverse=True)
for i in range(min(7, len(feat_imp))):
    print(" ", feat_imp[i][0], ":", round(feat_imp[i][1], 3))

# by game analysis
print("\n=== By Game ===")
games = df['game_name'].unique()
for g in games:
    sub = df[df['game_name'] == g]
    if len(sub) >= 10:
        avg = sub['total_time_seconds'].mean()
        corr = sub['max_steps'].corr(sub['total_time_seconds'])
        print(g + ": n=" + str(len(sub)) + " avg=" + str(round(avg,0)) + "s corr=" + str(round(corr,2)))

# cross val
print("\n=== CV ===")
scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print("mean:", round(scores.mean(), 3))
print("std:", round(scores.std(), 3))

# plots
fig = plt.figure(figsize=(10, 8))

# plot 1 - pred vs actual
ax1 = fig.add_subplot(2, 2, 1)
# find best model
best_pred = pred_lr
best_r2 = r2_lr
if r2_rf > best_r2:
    best_pred = pred_rf
    best_r2 = r2_rf
if r2_gb > best_r2:
    best_pred = pred_gb
    best_r2 = r2_gb
ax1.scatter(y_te, best_pred, alpha=0.5, s=15)
min_val = min(y_te.min(), min(best_pred))
max_val = max(y_te.max(), max(best_pred))
ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
ax1.set_xlabel('Actual (s)')
ax1.set_ylabel('Predicted (s)')
ax1.set_title('Pred vs Actual R2=' + str(round(best_r2, 3)))

# plot 2 - feature imp
ax2 = fig.add_subplot(2, 2, 2)
top_n = 8
names = []
vals = []
for i in range(min(top_n, len(feat_imp))):
    names.append(feat_imp[i][0])
    vals.append(feat_imp[i][1])
ax2.barh(names, vals)
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance')
ax2.invert_yaxis()

# plot 3 - time by game
ax3 = fig.add_subplot(2, 2, 3)
game_times = df.groupby('game_name')['total_time_seconds'].mean()
game_times = game_times.sort_values()
ax3.barh(game_times.index, game_times.values)
ax3.set_xlabel('Avg Time (s)')
ax3.set_title('Time by Game')

# plot 4 errors
ax4 = fig.add_subplot(2, 2, 4)
errors = []
for i in range(len(y_te)):
    errors.append(y_te.iloc[i] - best_pred[i])
ax4.hist(errors, bins=30, edgecolor='black')
ax4.axvline(0, color='r', linestyle='--')
ax4.set_xlabel('Error (s)')
ax4.set_title('Prediction Errors')

plt.tight_layout()
plt.savefig(cfg['output_plot'])
print("\nsaved plot to", cfg['output_plot'])

# save txt
f = open('results_summary.txt', 'w')
f.write("RQ1 Results\n")
f.write("===========\n\n")
f.write("Samples: " + str(len(X)) + "\n")
f.write("Features: " + str(len(feat)) + "\n\n")
f.write("Linear: R2=" + str(round(r2_lr,3)) + " MAE=" + str(round(mae_lr,0)) + "\n")
f.write("RF: R2=" + str(round(r2_rf,3)) + " MAE=" + str(round(mae_rf,0)) + "\n")
f.write("GradBoost: R2=" + str(round(r2_gb,3)) + " MAE=" + str(round(mae_gb,0)) + "\n\n")
f.write("Top features:\n")
for i in range(5):
    f.write("  " + feat_imp[i][0] + ": " + str(round(feat_imp[i][1],3)) + "\n")
f.close()
print("saved results_summary.txt")
