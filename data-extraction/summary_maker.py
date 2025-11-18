import pandas as pd




def summarize(target_run_id):

    df_full = pd.read_csv('E:/project2-1/data-collection/results/training.csv')

    df = df_full[df_full["RunID"] == target_run_id].reset_index(drop=True)

    total_duration_sec = df["episode_length"].sum()
    id_max_reward = df["cumulative_reward"].idxmax()
    time_to_max = df["episode_length"].iloc[:id_max_reward + 1].sum()

    result = {
        "run_id": target_run_id,
        "final_reward": df["cumulative_reward"].iloc[-1] if "cumulative_reward" in df else None,
        "max_reward": df["cumulative_reward"].max() if "cumulative_reward" in df else None,
        "time_to_max_reward_sec": time_to_max,
        "total_duration_sec":  total_duration_sec,

        "mean_policy_loss": df["policy_loss"].mean() if "policy_loss" in df else None,
        "mean_value_loss": df["loss_value"].mean() if "loss_value" in df else None,
        "mean_entropy": df["entropy"].mean() if "entropy" in df else None,
        "final_entropy": df["entropy"].iloc[-1] if "entropy" in df else None,

        "max_cpu_usage_percent": df["cpu_usage"].max() if "cpu_usage" in df else None,
        "mean_cpu_usage_percent": df["cpu_usage"].mean() if "cpu_usage" in df else None,

        "max_ram_used_mb": df["ram_used_mb"].max() if "ram_used_mb" in df else None,
        "mean_ram_used_mb": df["ram_used_mb"].mean() if "ram_used_mb" in df else None,

        "max_gpu_usage_percent": df["gpu_usage"].max() if "gpu_usage" in df else None,
        "mean_gpu_usage_percent": df["gpu_usage"].mean() if "gpu_usage" in df else None,

        "max_vram_used_mb": df["vram_used_mb"].max() if "vram_used_mb" in df else None,
        "mean_vram_used_mb": df["vram_used_mb"].mean() if "vram_used_mb" in df else None,
    }

    summary = pd.DataFrame([result])
    summary.to_csv('E:/project2-1/data-collection/results/summary.csv', index=False)

summarize("22/3DBall")
