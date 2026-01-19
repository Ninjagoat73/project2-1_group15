import pandas as pd


training_df = pd.read_csv('data-analysis/training_rows.csv')
static_df = pd.read_csv('data-analysis/static_rows.csv')
summary_df = pd.read_csv('data-analysis/summary_rows.csv')

# Merge static and training data

combined_df = pd.merge(training_df, static_df, on='RunID', how = 'left')

# Combine summary with static and training data

final_df = pd.merge(combined_df, summary_df, on = 'RunID' , how = 'left')

final_df.to_csv('input.csv', index = False)