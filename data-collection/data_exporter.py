import csv
import datetime
from hardware_info_getter import get_all_data

def create_run_id(training_type):
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{training_type}-{now_str}"
    return run_id

def create_static_csv(hardware_data: dict, run_id: str):

    header = [
        'run_id', 'os', 'cpu', 'physical_cores', 'system_ram_gb',
        'gpu', 'GPU_ram_mb', 'training_type', 'batch_size', 'learning_rate'
    ]


    filename = f"{run_id}-static.csv"

    data_dict = {
        'run_id': run_id,
        'os': hardware_data.get('os'),
        'cpu': hardware_data.get('cpu'),
        'physical_cores': hardware_data.get('physical_cores'),
        'system_ram_gb': hardware_data.get('system_ram_gb'),
        'gpu': hardware_data.get('gpu'),
        'GPU_ram_mb': hardware_data.get('GPU_ram_mb'),
        'training_type': "ppo",
        'batch_size': 1,
        'learning_rate': 2
    }

    try:

        with open(filename, 'w', newline='') as csvfile:

            csv_writer = csv.DictWriter(csvfile, fieldnames=header)


            csv_writer.writeheader()


            csv_writer.writerow(data_dict)

    except IOError as e:
        print(f"Error: Could not write to file {filename}. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_training_csv():
    pass

def create_summary_csv():
    pass


def create_all_csv():
    hardware_data = get_all_data()
    training_type = "ppo"
    run_id = create_run_id(training_type)
    create_static_csv(hardware_data, run_id)
