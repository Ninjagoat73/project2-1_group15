import yaml

def extract_static_yaml_value(yaml_path):
    with open(yaml_path, 'r') as f:
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
            print(f'Error reading data from {yaml_path}')
            pass

        return dict_data

    except (StopIteration,AttributeError) as e:
        print("This isnt supposed to happen")
        print(e)

if __name__ == '__main__':
    extract_static_yaml_value("../config/ppo/3DBall.yaml")
