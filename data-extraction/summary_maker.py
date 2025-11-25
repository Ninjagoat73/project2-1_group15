import csv
import os
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

header = [
    'RunID', 'final_reward',  'max_reward', 'time_to_max_reward_sec', 'total_duration_sec',

    'mean_policy_loss', 'mean_value_loss', 'mean_entropy', 'final_entropy',

    'max_cpu_usage_percent', 'mean_cpu_usage_percent', 'max_ram_used_mb', 'mean_ram_used_mb', 'mean_cpu_frequency', 'max_cpu_frequency',
    'max_gpu_usage_percent', 'mean_gpu_usage_percent', 'max_vram_used_mb',  'mean_vram_used_mb',
    'max_game_cpu_usage_percent', 'mean_game_cpu_usage_percent', 'max_game_memory_usage_mb',  'mean_game_memory_usage_mb'
]

def summarize( training_data: DataFrame, summary_data: DataFrame):

    try:
        success = True

        training_run_ids = training_data["RunID"].unique()
        summary_run_ids = summary_data["RunID"].unique()

        if training_data.empty:
            print(f"No training data. Please have training data ready.")
            sys.exit(0)

        for target_run_id in training_run_ids:

            if target_run_id in summary_run_ids:
                print(f"{target_run_id} already exists. Skipping.")
                continue

            df = training_data[training_data["RunID"] == target_run_id].reset_index(drop=True)

            result = {
                "RunID": target_run_id,

                "final_reward": df["cumulative_reward"].iloc[-1],
                "max_reward": df["cumulative_reward"].max(),

                "time_to_max_reward_sec": df["time_elapsed(s)"].iloc[df["cumulative_reward"].idxmax()],
                "total_duration_sec": df["time_elapsed(s)"].max(),

                "mean_policy_loss": df["policy_loss"].mean() ,
                "mean_value_loss": df["loss_value"].mean(),
                "mean_entropy": df["entropy"].mean() ,
                "final_entropy": df["entropy"].iloc[-1] ,

                "max_cpu_usage_percent": df["cpu_usage_percent"].max(),
                "mean_cpu_usage_percent": df["cpu_usage_percent"].mean(),

                "max_ram_used_mb": df["ram_usage_mb"].max(),
                "mean_ram_used_mb": df["ram_usage_mb"].mean(),

                "max_cpu_frequency": df["cpu_frequency_mhz"].max(),
                "mean_cpu_frequency": df["cpu_frequency_mhz"].mean(),

                "max_gpu_usage_percent": df["gpu_usage_percent"].max(),
                "mean_gpu_usage_percent": df["gpu_usage_percent"].mean() ,

                "max_vram_used_mb": df["vram_usage_mb"].max(),
                "mean_vram_used_mb": df["vram_usage_mb"].mean(),

                "mean_game_cpu_usage_percent": df["game_cpu_usage"].mean(),
                "max_game_cpu_usage_percent": df["game_cpu_usage"].max(),

                "mean_game_memory_usage_mb": df["game_memory_usage(MB)"].mean(),
                "max_game_memory_usage_mb": df["game_memory_usage(MB)"].max(),
            }


            new_df = pd.DataFrame([result])
            summary_data = pd.concat([summary_data, new_df], ignore_index=True)

        return success, summary_data
    except Exception as e:
        print("Error occurred while summarizing data")
        print(e)
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary_maker.py <path-to-training.csv>")
        sys.exit(0)

    training_csv = sys.argv[1]
    OUTPUT_FILENAME = 'summary.csv'

    if os.path.isfile(training_csv):
        print(f"{training_csv} file exists. Using data.")
        training_data = pd.read_csv(training_csv)
    else:
        print(f"{training_csv} was not found. Please have training data ready.")
        sys.exit(0)

    if os.path.isfile(OUTPUT_FILENAME):
        print(f"{OUTPUT_FILENAME} file already exists. Appending data to it...")
        summary_data = pd.read_csv(OUTPUT_FILENAME)
    else:
        print(f"{OUTPUT_FILENAME} was not found. Creating new {OUTPUT_FILENAME} file")
        summary_data = pd.DataFrame(columns=header)

    success, summary_data = summarize(training_data ,summary_data)

    if success:
        summary_data.to_csv(OUTPUT_FILENAME, index=False)
    else:
        print("Error occurred while summarizing data")


