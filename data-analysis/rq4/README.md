# RQ4 model
## Command-line usage
To run the script simply type this command in the terminal:
>python3 RandomForestRegression.py <path_to_config_file>

## Config file structure
The config file is a simple yaml file with following structure:
```yaml
rq4:
  random_state : 42
  n_estimators : 100
  max_depth : 15
  min_samples_leaf : 6
  max_features : 'sqrt'
  n_jobs : -1
  random_state_model : 42
  cv_folds : 10
```
