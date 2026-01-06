import pandas as pd

# 1. Load the Master List (static.csv)
print("Loading master static data...")
static_df = pd.read_csv('E:/project2-1/data-collection/results/static.csv')

# Get a unique list of valid RunIDs (this is our 'Golden List')
valid_run_ids = set(static_df['run_id'].unique())
print(f"Found {len(valid_run_ids)} valid RunIDs in static.csv")


def clean_large_file(file_path, valid_ids):
    print(f"Cleaning {file_path}...")
    # Load the target file
    df = pd.read_csv(file_path)
    initial_rows = len(df)

    # FILTER: Keep only rows where RunID is in our valid_ids set
    # Using 'isin' is extremely fast for this
    df_cleaned = df[df['RunID'].isin(valid_ids)]

    # Save it back (overwriting the original)
    df_cleaned.to_csv(file_path, index=False)

    removed = initial_rows - len(df_cleaned)
    print(f"Done. Removed {removed} orphan rows. New size: {len(df_cleaned)}")


# 2. Clean Training and Summary files
clean_large_file('E:/project2-1/data-collection/results/training.csv', valid_run_ids)
clean_large_file('E:/project2-1/data-collection/results/summary.csv', valid_run_ids)

print("\nAll files are now synchronized with static.csv!")
