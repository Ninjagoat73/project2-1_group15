import csv
from pathlib import Path

import pandas as pd

header = [
    'run_id', 'final_reward_mean', 'max_reward_mean', 'time_to_max_reward_sec', 'total_duration_sec',
    'mean_policy_loss', 'mean_value_loss', 'mean_entropy', 'batch_size', 'final_entropy', 'max_cpu_usage_percent',
    'mean_cpu_usage_percent',
    'max_ram_used_mb', 'mean_ram_used_mb', 'max_gpu_usage_percent', 'mean_gpu_usage_percent', 'max_vram_used_mb',
    'mean_vram_used_mb'
]




def summarize(target_run_id):

    success = True



    training_path = "E:/project2-1/data-collection/results/training.csv"
    summary_path = Path("summary.csv")
    df_full = pd.read_csv(training_path)

    df = df_full[df_full["RunID"] == target_run_id].reset_index(drop=True)

    print(df.head())

    result = {
        "run_id": target_run_id,
        "final_reward": df["cumulative_reward"].iloc[-1],
        "max_reward": df["cumulative_reward"].max() ,
        "time_to_max_reward_sec": None,
        "total_duration_sec":  None,

        "mean_policy_loss": df["policy_loss"].mean() ,
        "mean_value_loss": df["loss_value"].mean(),
        "mean_entropy": df["entropy"].mean() ,
        "final_entropy": df["entropy"].iloc[-1] ,

        "max_cpu_usage_percent": df["cpu_usage"].max(),
        "mean_cpu_usage_percent": df["cpu_usage"].mean(),

        "max_ram_used_mb": df["ram_used_mb"].max(),
        "mean_ram_used_mb": df["ram_used_mb"].mean(),

        "max_gpu_usage_percent": df["gpu_usage"].max() if "gpu_usage" in df else None,
        "mean_gpu_usage_percent": df["gpu_usage"].mean() if "gpu_usage" in df else None,

        "max_vram_used_mb": df["vram_used_mb"].max() if "vram_used_mb" in df else None,
        "mean_vram_used_mb": df["vram_used_mb"].mean() if "vram_used_mb" in df else None,
    }

    try:
        if summary_path.is_file():
            with open(summary_path, 'a', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writerow(result)
        else:
            with open(summary_path, 'w', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writeheader()
                csv_writer.writerow(result)
    except IOError as e:
        print(f"Error: Could not write to file {summary_path}. {e}")
        success = False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        success = False

    return success

summarize("first_run/3DBall")
