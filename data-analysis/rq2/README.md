# RQ2 model
## Command-line usage
To run the script simply type this command in the terminal:
>python3 rq2.py <path_to_config_file>

## Config file structure
The config file is a simple yaml file with following structure:
```yaml
rq2:
  random_state : 42
  randomforest:
    n_estimators : 300
    forest_random_state : 42
    n_jobs : -1
  gradientboosting:
    max_depth : 6
    learning_rate : 0.05
    boosting_random_state : 42
```
