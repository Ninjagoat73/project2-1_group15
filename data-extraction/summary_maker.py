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




def summarize( target_run_id: str, training_data: DataFrame, summary_data: DataFrame):



    try:
        success = True

        df_full = training_data
        df = df_full[training_data["RunID"] == target_run_id].reset_index(drop=True)

        if df.empty:
            print(f"{target_run_id} was not found. Please have training data ready.")
            sys.exit(0)

        if target_run_id in summary_data["RunID"].values:
            print(f"{target_run_id} was found in summary data. Exiting.")
            sys.exit(0)

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
    if len(sys.argv) < 3:
        print("Usage: python summary_maker.py <path-to-results-folder> <run-id>")
        sys.exit(0)

    folder = sys.argv[1]
    run_id = sys.argv[2]
    OUTPUT_FILENAME = 'summary.csv'
    INPUT_FILENAME = 'training.csv'

    OUTPUT_FILEPATH = folder + "/" + OUTPUT_FILENAME
    INPUT_FILEPATH = folder + "/" + INPUT_FILENAME

    if not os.path.isdir(folder):
        print(f"\n Fatal Error: root '{folder}' does not exist.")
        exit()

    if os.path.isfile(INPUT_FILEPATH):
        print(f"{INPUT_FILEPATH} file exists. Using data.")
        training_data = pd.read_csv(INPUT_FILEPATH)
    else:
        print(f"{INPUT_FILEPATH} was not found. Please have training data ready.")
        sys.exit(0)


    if os.path.isfile(OUTPUT_FILEPATH):
        print(f"{OUTPUT_FILEPATH} file already exists. Appending data to it...")
        summary_data = pd.read_csv(OUTPUT_FILEPATH)
    else:
        print(f"{OUTPUT_FILEPATH} was not found. Creating new {OUTPUT_FILEPATH} file")
        summary_data = pd.DataFrame(columns=header)

    success, summary_data = summarize(run_id, training_data ,summary_data)



    if success:
        summary_data.to_csv(OUTPUT_FILEPATH, index=False)
    else:
        print("Error occurred while summarizing data")


