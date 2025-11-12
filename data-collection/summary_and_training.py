def create_training_csv(run_id: str, hardware_data: dict):
    header = [
        'run_id', 'cpu_frequency_mhz', 'cpu_usage_percent', 'gpu_usage_percent', 'vram_usage_mb',
        'ram_usage_mb', 'episode', 'step', 'agent_id', 'reward', 'stability_reward', 'loss_value', 'entropy', 'policy_loss', 'value_estimate'
        'cumulative_reward', 'reward_mean_100', 'time_elapsed_ms', 'steps_per_second'
    ]

    filename = f"{run_id}-training.csv"

    data_dict = {
    }

def create_summary_csv(run_id: str, hardware_data: dict ):
    header = [
        'run_id', 'final_reward_mean', 'max_reward_mean', 'time_to_max_reward_sec', 'total_duration_sec',
        'mean_policy_loss', 'mean_value_loss', 'mean_entropy', 'batch_size', 'final_entropy', 'max_cpu_usage_percent', 'mean_cpu_usage_percent',
        'max_ram_used_mb', 'mean_ram_used_mb', 'max_gpu_usage_percent', 'mean_gpu_usage_percent', 'max_vram_used_mb', 'mean_vram_used_mb'
    ]
    filename = f"{run_id}-summary.csv"
