# merge static and training data for ML

import pandas as pd

# load files
static = pd.read_csv("static.csv")
training = pd.read_csv("training.csv")

print("loaded static:", len(static))
print("loaded training:", len(training))

# get total time per run from training data
# training has many rows per run so we take the max time
times = training.groupby('RunID')['time_elapsed(s)'].max()
times = times.reset_index()
times.columns = ['run_id', 'total_time_seconds']

# run_id in training has format like "20251123-014754/3DBall"
# we need just the first part to match with static
times['run_id'] = times['run_id'].str.split('/').str[0]

# merge the two dataframes
data = static.merge(times, on='run_id', how='inner')

print("merged:", len(data))

# convert ram from bytes to gb
data['ram_gb'] = data['system_ram_gb'] / (1024**3)

# save
data.to_csv("training_data_prepared.csv", index=False)
print("saved to training_data_prepared.csv")
