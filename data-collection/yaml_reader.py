import yaml

header = [
    'run_id', 'os', 'cpu', 'physical_cores', 'system_ram_gb',
    'gpu', 'GPU_ram_mb', 'cuda_version', 'training_type', 'batch_size',
    'learning_rate', 'buffer_size', 'beta', 'episolon', 'lambd',
    'num_epochs', 'learning_rate_schedule', 'normalize', 'hidden_units',
    'num_layers', 'vis_encode_type', 'reward_extrinsic_gamma',
    'reward_extrinsic_strength', 'keep_checkpoints', 'max_step',
    'time_horizon', 'summary_freq', 'sequence_length', 'memory_size',
    'curiosity_gamma', 'curiosity_strenght', 'curiosity_hidden_units',
    'curiosity_learning_rate', 'save_steps', 'team_change', 'swap_steps',
    'window', 'play_againts_latest_model_ratio', 'initial_elo'
]

def extract_static_yaml_value(yaml_data: dict):

    try:
        layer1 = next(iter(yaml_data.values()))
        data = next(iter(layer1.values()))


        dict_data = {}


        training_type = data['trainer_type']
        for key, value in data['hyperparameters'].items():
            dict_data.update({key: value})
            print(key, value)
        for key, value in data['network_settings'].items():
            dict_data.update({key: value})


    except (StopIteration,AttributeError) as e:
        print("This isnt supposed to happen")
        print(e)



with open("E:/project2-1/config/ppo/3DBall.yaml") as file:
    try:
        extracted_data = yaml.safe_load(file)
        cleaned_data = extract_static_yaml_value(extracted_data)
    except yaml.YAMLError as exc:
        print(exc)
