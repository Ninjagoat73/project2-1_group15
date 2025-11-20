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
            'epsilon': yaml_data.get('epsilon'),
            'lambd': yaml_data.get('lambd'),
            'num_epoch': yaml_data.get('num_epoch'),
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
            'summary_freq': yaml_data.get('summary_freq')
        }

header = [
        'run_id', 'os', 'cpu', 'physical_cores', 'system_ram_gb',
        'gpu', 'GPU_ram_mb', 'cuda_version', 'training_type', 'game_name', 'batch_size',
        'learning_rate', 'buffer_size', 'beta', 'epsilon', 'lambd',
        'num_epoch', 'learning_rate_schedule', 'normalize', 'hidden_units',
        'num_layers', 'vis_encode_type', 'reward_extrinsic_gamma',
        'reward_extrinsic_strength', 'keep_checkpoints', 'max_steps',
        'time_horizon', 'summary_freq'
    ]
