import pandas as pd


data_path = "E:/project2-1\data-collection\data/"


static_df = pd.read_csv(data_path + 'static.csv')
summary_df = pd.read_csv(data_path + 'summary.csv')


merged_df = pd.merge(
    static_df[['RunID', 'game_name']],
    summary_df[['RunID', 'final_reward']],
    on='RunID'
)

pushblock_df = merged_df[merged_df['game_name'] == 'Walker'].copy()
pushblock_df = pushblock_df.sort_values(by='final_reward', ascending=False)

run_reward_dict = {
    f"{row['game_name']}-{row['RunID']}": row['final_reward']
    for _, row in pushblock_df.iterrows()
}

min = pushblock_df['final_reward'].min()
max = pushblock_df['final_reward'].max()

print("Sample Output:")
for key in list(run_reward_dict.keys())[:5]:
    print(f'"{key}": {run_reward_dict[key]}')
output_df = pd.DataFrame(list(run_reward_dict.items()), columns=['Run_Identifier', 'Final_Reward'])
print("\nDataFrame Preview:")
print(output_df.head())
print(min)
print(max)
output_df.to_csv(data_path + 'test.csv', index=False)
