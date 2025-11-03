import yaml

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

def extract_static_yaml_value(filename):

    with open(filename, 'r') as f:
        full_yaml = yaml.safe_load(f)

    try:
        layer1 = next(iter(full_yaml.values()))
        data = next(iter(layer1.values()))


        dict_data = {}


        dict_data.update({'training_type': data['trainer_type']})
        dict_data.update({'keep_checkpoints': data['keep_checkpoints']})
        dict_data.update({'max_steps': data['max_steps']})
        dict_data.update({'time_horizon': data['time_horizon']})
        dict_data.update({'summary_freq': data['summary_freq']})
        for key, value in data['hyperparameters'].items():
            dict_data.update({key: value})
        for key, value in data['network_settings'].items():
            dict_data.update({key: value})
        for key, value in data['reward_signals']['extrinsic'].items():
            dict_data.update({key: value})

        try:
            for key, value in data['reward_signals']['curiosity'].items():
                curiosity_key = 'curiosity_' + key
                dict_data.update({curiosity_key: value})
            for key, value in data['self_play'].items():
                dict_data.update({key: value})
        except:
            pass


        return dict_data



    except (StopIteration,AttributeError) as e:
        print("This isnt supposed to happen")
        print(e)


if __name__ == '__main__':
    extract_static_yaml_value("E:/project2-1/config/ppo/3DBall.yaml")

