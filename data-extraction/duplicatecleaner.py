import pandas as pd

# 1. Load the data
file_path = 'E:/project2-1/data-collection/static.csv'
print(f"Loading {file_path}...")
df = pd.read_csv(file_path)

# Record initial count for comparison
initial_count = len(df)

# 2. Drop duplicates based on the RunID column
# keep='first' means it keeps the first occurrence and deletes the rest
df_cleaned = df.drop_duplicates(subset=['run_id'], keep='first')

# 3. Save the cleaned file
df_cleaned.to_csv(file_path, index=False)

# 4. Report results
final_count = len(df_cleaned)
removed = initial_count - final_count
print(f"Done! Removed {removed} duplicate rows.")
print(f"Total unique runs remaining: {final_count}")
