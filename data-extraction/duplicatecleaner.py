import pandas as pd


file_path = 'E:/project2-1/data-collection/static.csv'
print(f"Loading {file_path}...")
df = pd.read_csv(file_path)


initial_count = len(df)


df_cleaned = df.drop_duplicates(subset=['run_id'], keep='first')


df_cleaned.to_csv(file_path, index=False)

final_count = len(df_cleaned)
removed = initial_count - final_count
print(f"Done! Removed {removed} duplicate rows.")
print(f"Total unique runs remaining: {final_count}")
