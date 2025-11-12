def create_static_dict(run_id: str, hardware_data: dict, yaml_data: dict):
    """
        This function simply combines the hardware data and yaml data into a single dictionary
        Returns: dictionary containing an entry for the static.csv file
    """
    return {
            'run_id': run_id,
            'os': hardware_data.get('os'),
            'cpu': hardware_data.get('cpu'),
            'physical_cores': hardware_data.get('physical_cores'),
            'system_ram_gb': hardware_data.get('system_ram_gb'),
            'gpu': hardware_data.get('gpu'),
            'GPU_ram_mb': hardware_data.get('GPU_ram_mb'),
            'training_type': yaml_data.get('training_type'),
            'game_name': yaml_data.get('game_name'),
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
           # 'sequence_length': yaml_data.get('sequence_length'),
           # 'memory_size': yaml_data.get('memory_size'),
           # 'curiosity_gamma': yaml_data.get('curiosity_gamma'),
           # 'curiosity_strength': yaml_data.get('curiosity_strength'),
           # 'curiosity_hidden_units': yaml_data.get('curiosity_hidden_units'),
           # 'curiosity_learning_rate': yaml_data.get('curiosity_learning_rate'),
           # 'save_steps': yaml_data.get('save_steps'),
           # 'team_change': yaml_data.get('team_change'),
           # 'swap_steps': yaml_data.get('swap_steps'),
           # 'window': yaml_data.get('window'),
           # 'play_againts_latest_model_ratio': yaml_data.get('play_againts_latest_model_ratio'),
           # 'initial_elo': yaml_data.get('initial_elo')
        }

header = [
        'run_id', 'os', 'cpu', 'physical_cores', 'system_ram_gb',
        'gpu', 'GPU_ram_mb', 'cuda_version', 'training_type', 'game_name', 'batch_size',
        'learning_rate', 'buffer_size', 'beta', 'episolon', 'lambd',
        'num_epochs', 'learning_rate_schedule', 'normalize', 'hidden_units',
        'num_layers', 'vis_encode_type', 'reward_extrinsic_gamma',
        'reward_extrinsic_strength', 'keep_checkpoints', 'max_steps',
        'time_horizon', 'summary_freq'
       # 'sequence_length', 'memory_size',
       # 'curiosity_gamma', 'curiosity_strength', 'curiosity_hidden_units',
       # 'curiosity_learning_rate', 'save_steps', 'team_change', 'swap_steps',
       # 'window', 'play_againts_latest_model_ratio', 'initial_elo'
    ]


