import yaml
def extract_batch_config_file(yaml_path : str) -> dict:
    """_summary_

    Args:
        yaml_path (str): Path to batch configuration file that will be used to run the trainings runs

    Returns:
        dict: Batch configuration with keys:
            - batch_size (int): Size of the batch
            - algorithms (list[str]): Enabled algorithm names
            - games (list[str]): Enabled game names
    
    Expected YAML format (look at test_batch_config.yaml):
        
        batch_config:
            batch_size: 1
            algorithms:
                ppo: true
                sac: false
            games:
                3DBall: true
                GridWorld: true
                Walker: false
    
    Note:
        Only algorithms and games with value 'true' will be included
        in the returned lists.
    """
    dict_data = {}
    with open(yaml_path, 'r') as f:
        try:
            full_yaml = yaml.safe_load(f)
        except:
            print(f"Error reading {yaml_path}")
            return {}
    try:
        batch_config = full_yaml.get("batch_config", {})
        dict_data["batch_size"] = batch_config["batch_size"]
        algorithms = []
        for algorithm, value in batch_config["algorithms"].items():
            if value:
                algorithms.append(algorithm)
        dict_data["algorithms"] = algorithms
        games = []
        for game, value in batch_config["games"].items():
            if value:
                games.append(game)
        dict_data["games"] = games
    
    except:
        print("There was an issue with processing batch configuration file. Default values will be used")
        dict_data = generate_default_config_dict()
    finally:
        print(dict_data)
        return dict_data

def generate_default_config_dict():
    return {'batch_size': 5, 'algorithms': ['ppo', 'sac'], 'games': ['3DBall', '3DBallHard', 'Crawler', 'GridFoodCollector', 'GridWorld', 'Hallway', 'PushBlock', 'Pyramids', 'Sorter', 'Walker', 'Worm']}
if __name__ == "__main__":
    extract_batch_config_file("test_batch_config.yaml")