import data_exporter
import yaml_reader
import os
import sys
import datetime
import subprocess

def run_training(yaml_path: str, run_id: str, executable_path: str):
    result = subprocess.run([
        "mlagents-learn",
        yaml_path,
        "--env=" + executable_path,
        "--run-id=" + run_id,
#        "--no-graphics"
    ])
    if result.returncode != 0:
        print("Training failed. No entry will be added to static.csv")
        return
    data_exporter.create_static_csv(run_id, yaml_path)

if len(sys.argv) < 3:
    print("Usage: python3 training_scipt.py <path-to-yaml-folder> <path-to-executable-folder>")
    sys.exit(0)

yaml_folder_path = sys.argv[1]
executable_folder_path = sys.argv[2]

if not os.path.isdir(yaml_folder_path):
    print(f"Error: '{yaml_folder_path}' is not a valid directory.")
    sys.exit(0)

if not os.path.isdir(executable_folder_path):
    print(f"Error: '{executable_folder_path}' is not a valid directory.")
    sys.exit(0)

for yaml in os.listdir(yaml_folder_path):
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    yaml_path = os.path.join(yaml_folder_path, yaml)
    if os.path.isfile(yaml_path):
        if not yaml_path.endswith(".yaml"):
            print(f"{yaml_path} is not a yaml file. Skipping...")
            continue
        print(f"File being parsed: {yaml_path}")
        yaml_dict = yaml_reader.extract_static_yaml_value(yaml_path)
        game_name = yaml_dict['game_name']
        executable_path = executable_folder_path + "/" + game_name + ".app"
        print("executable path;", executable_path)
        run_training(yaml_path, run_id, executable_path)
