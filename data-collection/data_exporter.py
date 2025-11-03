import csv
import datetime
from hardware_info_getter import get_all_data
from yaml_reader import extract_static_yaml_value

def create_run_id(training_type):
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{training_type}-{now_str}"
    return run_id

def create_static_csv(run_id: str, hardware_data: dict, yaml_data: dict ):


    print(hardware_data)
    print(yaml_data)

    header = [
        'run_id', 'os', 'cpu', 'physical_cores', 'system_ram_gb',
        'gpu', 'GPU_ram_mb', 'cuda_version', 'training_type', 'batch_size',
        'learning_rate', 'buffer_size', 'beta', 'episolon', 'lambd',
        'num_epochs', 'learning_rate_schedule', 'normalize', 'hidden_units',
        'num_layers', 'vis_encode_type', 'reward_extrinsic_gamma',
        'reward_extrinsic_strength', 'keep_checkpoints', 'max_steps',
        'time_horizon', 'summary_freq', 'sequence_length', 'memory_size',
        'curiosity_gamma', 'curiosity_strength', 'curiosity_hidden_units',
        'curiosity_learning_rate', 'save_steps', 'team_change', 'swap_steps',
        'window', 'play_againts_latest_model_ratio', 'initial_elo'
    ]


    filename = f"{run_id}-static.csv"

    data_dict = {
        'run_id': run_id,
        'os': hardware_data.get('os'),
        'cpu': hardware_data.get('cpu'),
        'physical_cores': hardware_data.get('physical_cores'),
        'system_ram_gb': hardware_data.get('system_ram_gb'),
        'gpu': hardware_data.get('gpu'),
        'GPU_ram_mb': hardware_data.get('GPU_ram_mb'),
        'training_type': yaml_data.get('training_type'),
        'batch_size': yaml_data.get('batch_size'),
        'learning_rate': yaml_data.get('learning_rate'),
        'buffer_size': yaml_data.get('buffer_size'),
        'beta': yaml_data.get('beta'),
        'episolon': yaml_data.get('episolon'),
        'lambd': yaml_data.get('lambd'),
        'num_epochs': yaml_data.get('num_epochs'),
        'learning_rate_schedule': yaml_data.get('learning_rate_schedule'),
        'normalize': yaml_data.get('normalize'),
        'hidden_units': yaml_data.get('hidden_units'),
        'num_layers': yaml_data.get('num_layers'),
        'vis_encode_type': yaml_data.get('vis_encode_type'),
        'reward_extrinsic_gamma': yaml_data.get('reward_extrinsic_gamma'),
        'reward_extrinsic_strength': yaml_data.get('reward_extrinsic_strength'),
        'keep_checkpoints': yaml_data.get('keep_checkpoints'),
        'max_steps': yaml_data.get('max_steps'),
        'time_horizon': yaml_data.get('time_horizon'),
        'summary_freq': yaml_data.get('summary_freq'),
        'sequence_length': yaml_data.get('sequence_length'),
        'memory_size': yaml_data.get('memory_size'),
        'curiosity_gamma': yaml_data.get('curiosity_gamma'),
        'curiosity_strength': yaml_data.get('curiosity_strength'),
        'curiosity_hidden_units': yaml_data.get('curiosity_hidden_units'),
        'curiosity_learning_rate': yaml_data.get('curiosity_learning_rate'),
        'save_steps': yaml_data.get('save_steps'),
        'team_change': yaml_data.get('team_change'),
        'swap_steps': yaml_data.get('swap_steps'),
        'window': yaml_data.get('window'),
        'play_againts_latest_model_ratio': yaml_data.get('play_againts_latest_model_ratio'),
        'initial_elo': yaml_data.get('initial_elo')
    }

    try:

        with open(filename, 'w', newline='') as csvfile:

            csv_writer = csv.DictWriter(csvfile, fieldnames=header)


            csv_writer.writeheader()


            csv_writer.writerow(data_dict)

    except IOError as e:
        print(f"Error: Could not write to file {filename}. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

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


def create_all_csv():
    hardware_data = get_all_data()
    yaml_data = extract_static_yaml_value("E:/project2-1/config/ppo/3DBall.yaml")
    training_type = yaml_data['training_type']
    run_id = create_run_id(training_type)
    create_static_csv(run_id, hardware_data, yaml_data)


create_all_csv()
