import os
import pandas as pd
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

# Creating tag map to match data from tensorboard to columns in csv file
TAG_MAP = {

    'Environment/Cumulative Reward': 'cumulative_reward',
    'Environment/Episode Length': 'episode_length',

    'Losses/Policy Loss': 'policy_loss',
    'Losses/Value Loss': 'loss_value',

    'Policy/Entropy': 'entropy',
    'Policy/Extrinsic Value Estimate': 'extrinsic_value_estimate',
    'Policy/Beta' : 'beta',
    'Policy/Epsilon' : 'epsilon',
    'Policy/Extrinsic Reward' : 'extrinsic_reward',
    'Policy/Learning Rate' : 'learning_rate',

    'Performance/cpuUsagePercent' : 'cpu_usage_percent',
    'Performance/cpuFrequency' : 'cpu_frequency_mhz',
    'Performance/gameCpuUsage(%)' : 'game_cpu_usage',
    'Performance/gameMemoryUsage(MB)' : 'game_memory_usage(MB)',
    'Performance/ramUsageMB' : 'ram_usage_mb',
    'Performance/gpuUsagePercent' : 'gpu_usage_percent',
    'Performance/vramUsageMB' : 'vram_usage_mb',
    'Performance/episodeTime(s)' : 'episode_time',
    'Performance/stepLength(ms)' : 'step_length(ms)',
    'Performance/stepsPerSecond': 'steps_per_second',
    'Performance/timeElapsed(s)' : 'time_elapsed(s)',
}


def get_log_directories(folder):
    # Function that finds all the directories with event file of training in the root folder.
    log_dirs = []

    # Go through all the folders
    run_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    if not run_folders:
        print(f" No run folders found in: {folder}")
        return []

    for run_folder in run_folders:
        run_path = os.path.join(folder, run_folder)

        # Recursive search for event file in the run_id folder
        event_files = glob(os.path.join(run_path, '**', 'events.out.tfevents*'), recursive = True)

        if event_files:
            log_dir = os.path.dirname(event_files[0])
            relative_log_dir = os.path.relpath(log_dir, folder).replace(os.sep, '/')
            log_dirs.append({
                'log_dir': log_dir,
                'run_id': relative_log_dir
            })
            print(f" Found logs for: {relative_log_dir}")
        else:
            print(f" No event files found in run folder: {run_folder}")

    return log_dirs

def extract_data_from_log(log_dir_info):
    log_dir = log_dir_info['log_dir']
    run_id = log_dir_info['run_id']
    all_data = []

    try:
        event_acc = EventAccumulator(log_dir, size_guidance = {'scalars': 0})
        event_acc.Reload()

        for tag in event_acc.Tags().get('scalars', []):
            if tag not in TAG_MAP:
                continue
            column_name = TAG_MAP[tag]
            events = event_acc.Scalars(tag)
            tag_data = [(e.step, e.value) for e in events]

            data_frame = pd.DataFrame(tag_data, columns = ['Step', 'MetricValue'])
            data_frame['ColumnName'] = column_name
            data_frame['Tag'] = tag
            data_frame['DataType'] = 'Scalar'
            all_data.append(data_frame)

    except Exception as e:
        print(f" Failed to process logs for {run_id}. Error: {e}")
        return pd.DataFrame()

    if all_data:
        data_frame_run = pd.concat(all_data, ignore_index = True)
        data_frame_run['RunID'] = run_id
        return data_frame_run

    return pd.DataFrame()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_mlagents_data.py <path-to-results-folder>")
        sys.exit(0)

    folder = sys.argv[1]
    OUTPUT_FILENAME = 'training.csv'

    if not os.path.isdir(folder):
        print(f"\n Fatal Error: root '{folder}' does not exist.")
        exit()

    if os.path.isfile(OUTPUT_FILENAME):
        print(f"{OUTPUT_FILENAME} file already exists. Appending data to it...")
        training_data = pd.read_csv(OUTPUT_FILENAME)
    else:
        print(f"{OUTPUT_FILENAME} was not found. Creating new {OUTPUT_FILENAME} file")
        training_data = pd.DataFrame(columns=["RunID", "Step"])

    log_directories = get_log_directories(folder)

    if not log_directories:
        print("No log directories found. Exiting.")
        exit()

    for log_info in log_directories:
        data_frame_run = extract_data_from_log(log_info)
        if not data_frame_run.empty:
            if log_info['run_id'] in training_data['RunID'].values:
                print(f"run_id {log_info['run_id']} already detected. Skipping this entry...")
                continue
            data_frame_run = data_frame_run.pivot(index=['RunID', 'Step'], columns='ColumnName', values='MetricValue')
            data_frame_run = data_frame_run.reset_index()
            training_data = pd.concat([training_data, data_frame_run], ignore_index=True)

            print(f"Extracted {len(data_frame_run)} total data points from {log_info['run_id']}")
        else:
            print(f"Data extraction for {log_info['run_id']} failed")

    if not training_data.empty:
        training_data.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n Extraction Complete")
        print(f"Total metrics extracted: {len(training_data)}")
        print(f"Output saved to: {OUTPUT_FILENAME}")
    else:
        print("\n Extraction Failed ")
