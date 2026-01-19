import pandas as pd


print("Loading master static data...")
static_df = pd.read_csv('../data-collection/data/static.csv')

valid_run_ids = set(static_df['RunID'].unique())
print(f"Found {len(valid_run_ids)} valid RunIDs in static.csv")


def clean_large_file(file_path, valid_ids):
    print(f"Cleaning {file_path}...")

    df = pd.read_csv(file_path)
    initial_rows = len(df)


    df_cleaned = df[df['RunID'].isin(valid_ids)]


    df_cleaned.to_csv(file_path, index=False)

    removed = initial_rows - len(df_cleaned)
    print(f"Done. Removed {removed} orphan rows. New size: {len(df_cleaned)}")

clean_large_file('../data-collection/data/training.csv', valid_run_ids)
clean_large_file('../data-collection/data/summary.csv', valid_run_ids)
print("\nAll files are now synchronized with static.csv!")
