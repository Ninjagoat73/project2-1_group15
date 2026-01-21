# RQ3 model
## Command-line usage
Since we have two models, we can have to command line commands:
>python per_game_model.py <path_to_config_file>
> 
>python global_model.py <path_to_config_file>

## Config file structure
The config file is a simple yaml file with following structure:
```yaml
random_forest:
 random_state: 1
 n_estimators: 100
 max_depth: 10
 min_samples_leaf: 5
 test_size: 0.2
 data_path: "../../data-collection/data"
```

note that the data path folder should contain summary.csv and static.csv
