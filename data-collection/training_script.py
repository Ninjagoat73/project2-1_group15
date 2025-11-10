import data_exporter
import os
import sys
import datetime
import subprocess

def run_training(yaml_path: str, run_id: str, executable_path: str):
    result = subprocess.run([
        "mlagents-learn",
        "/Users/pedropianna/project2-1_group15/data-collection/test/3DBall.yaml",
        "--env=" + executable_path, # TODO: extract the type from yaml file
        "--run-id=" + run_id,
        "--no-graphics"
    ])
    if result.returncode != 0:
        print("Training failed. No entry will be added to static.csv")
        return
    data_exporter.create_static_csv(run_id, yaml_path)

if len(sys.argv) < 3:
    print("Usage: python3 training_scipt.py <path-to-yaml-folder> <path-to-executable>")
    sys.exit(0)

folder_path = sys.argv[1]
executable_path = sys.argv[2]

if not os.path.isdir(folder_path):
    print(f"Error: '{folder_path}' is not a valid directory.")
    sys.exit(0)

for yaml in os.listdir(folder_path):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    yaml_path = os.path.join(folder_path, yaml)
    if os.path.isfile(yaml_path):
        run_training(yaml_path, run_id, executable_path)
