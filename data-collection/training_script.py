import data_exporter
import yaml_reader
import os
import sys
import datetime
import subprocess
import yaml_generator
import platform
import sys
import extract_mlagents_data
import summary_maker


def run_training(yaml_path: str, run_id: str, executable_path: str, results_path: str, data_path:str):
    result = subprocess.run([
        "mlagents-learn",
        yaml_path,
        "--env=" + executable_path,
        "--run-id=" + run_id,
        "--no-graphics"
    ])
    if result.returncode != 0:
        print("Training failed. No entry will be added to static.csv")
        return
    print("Static appending started...")
    data_exporter.create_static_csv(run_id, yaml_path)

    training_csv = os.path.join(data_path, "training.csv")
    summary_csv = os.path.join(data_path, "summary.csv")

    print("Training appending started...")
    extract_mlagents_data.extract_single_run(results_path, run_id, training_csv)

    print("Summary appending started...")
    summary_maker.summarize_single_run(run_id,training_csv, summary_csv)


if len(sys.argv) < 2:
    print("Usage: python3 training_scipt.py <path-to-executable-folder> <path-to-results-folder> <path-to-data-folder>")
    sys.exit(0)

games = ['Crawler','GridFoodCollector','Hallway','PushBlock','Pyramids','Sorter','Walker','Worm']
# games = ['3DBall','3DBallHard', 'Crawler','GridFoodCollector','GridWorld','Hallway','PushBlock','Pyramids','Sorter','Walker','Worm']
yamls = yaml_generator.generate_batch(count_per_algorithm=5, behaviours=games)
executable_folder_path = sys.argv[1]
results_folder_path = sys.argv[2]
data_path = sys.argv[3]
os_name = platform.system()

if not os.path.isdir(executable_folder_path):
    print(f"Error: '{executable_folder_path}' is not a valid directory.")
    sys.exit(0)

for success, yaml in yamls:
    if not success:
        continue
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # yaml_path = os.path.join(yaml_folder_path, yaml)
    if os.path.isfile(yaml):
        if not yaml.endswith(".yaml"):
            print(f"{yaml} is not a yaml file. Skipping...")
            continue
        print(f"File being parsed: {yaml}")
        yaml_dict = yaml_reader.extract_static_yaml_value(yaml)
        game_name = yaml_dict['game_name']
        if os_name == "Darwin":
            executable_path = executable_folder_path + "/" + game_name + ".app"
        else:
            executable_path = executable_folder_path + "\\" + game_name
        print("executable path;", executable_path)
        run_training(yaml, run_id, executable_path, results_folder_path, data_path)
