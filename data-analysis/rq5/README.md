# RQ5 model
## Command-line usage
To run the script simply type this command in the terminal:
>python clustering.py <path_to_config_file>

## Config file structure
The config file is a simple yaml file with following structure:
```yaml
clustering:
  subsample_cap : 5
  random_state : 42
  csv_file_path : "summed_and_filtered.csv"
  kmeans:
    k_min : 2
    k_max : 20
    k_final : 4
  dbscan:
    k_dist : 5
    eps_min : 0.2
    eps_max : 0.8
    eps_interval : 0.1
    eps_final : 0.4
```

Note that the csv file has to already be cleaned of NaNs and invalid rows
