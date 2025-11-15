import os
import pandas as pd
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT_LOG_DIR = './results'

# Creating tag map to match data from tensorboard to columns in csv file
TAG_MAP = {

    'Environment/Cumulative Reward': 'cumulative_reward',
    'Environment/Episode Length': 'episode_length',

    'Losses/Policy Loss': 'policy_loss',
    'Losses/Value Loss': 'loss_value',
    'Policy/Entropy': 'entropy',
    'Policy/Extrinsic Value Estimate': 'value_estimate',

    'Custom/Reward_Mean_100': 'reward_mean_100',
    'Custom/Reward': 'reward', 
    'Custom/Time_Elapsed_ms': 'time_elapsed_ms',
    'Custom/Steps_Per_Second': 'steps_per_second',
}


def get_log_directories():
    # Function that finds all the directories with event file of training in the root folder.
    
    log_dirs = []


    # Go through all the folders
    run_folders = [d for d in os.listdir(ROOT_LOG_DIR) if os.path.isdir(os.path.join(ROOT_LOG_DIR, d))]


    if not run_folders:
        print(f" No run folders found in: {ROOT_LOG_DIR}")
        return []
    
    for run_folder in run_folders:

        run_path = os.path.join(ROOT_LOG_DIR, run_folder)

        # Recursive search for event file in the run_id folder
        event_files = glob(os.path.join(run_path, '**', 'events.out.tfevents*'), recursive = True)

        if event_files:

            log_dir = os.path.dirname(event_files[0])

            relative_log_dir = os.path.relpath(log_dir, ROOT_LOG_DIR).replace(os.sep, '/')

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


    OUTPUT_FILENAME = 'training.csv'


    if not os.path.isdir(ROOT_LOG_DIR):
        print(f"\n Fatal Error: root '{ROOT_LOG_DIR}' does not exist.")
        exit()

    log_directories = get_log_directories()

    if not log_directories:
        print("No log directories found. Exiting.")
        exit()

    all_runs_data = []

    for log_info in log_directories:
        
        data_frame_run = extract_data_from_log(log_info)
        if not data_frame_run.empty:
            all_runs_data.append(data_frame_run)
            print(f" Extracted {len(data_frame_run)} total data points from {log_info['run_id']}")
    

    if all_runs_data:

        final_data = pd.concat(all_runs_data, ignore_index=True)
        
        # Formating for csv file so that all the column values are in the same row
        final_data = final_data.pivot(index=['RunID', 'Step'], columns='ColumnName', values='MetricValue')

        final_data = final_data.reset_index()

        final_data.to_csv(OUTPUT_FILENAME, index=False)
        
        print(f"\n Extraction Complete")
        print(f"Total metrics extracted: {len(final_data)}")
        print(f"Output saved to: {OUTPUT_FILENAME}")
    else:
        print("\n Extraction Failed ")
        print("No data was successfully read from any log file.")