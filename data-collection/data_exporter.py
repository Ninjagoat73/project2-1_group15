import csv
from pathlib import Path
from hardware_info_getter import get_all_data
from yaml_reader import extract_static_yaml_value
from static_dict import create_static_dict, header

def create_static_csv(run_id: str, yaml_path: str):
    """
        This function takes a run_id and the path of a YAML file as parameters and adds that information, as well as hardware data, as an entry to the static.csv file
        Returns: boolean stating if operation was successful
    """
    success = True
    yaml_data = extract_static_yaml_value(yaml_path)
    hardware_data = get_all_data()
    print("Hardware data to be added to entry: ", hardware_data)
    print("YAML data to be added to entry:", yaml_data)

    data_dict = create_static_dict(run_id, hardware_data, yaml_data)
    filename = Path("static.csv")
    try:
        if filename.is_file():
            with open(filename, 'a', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writerow(data_dict)
        else:
             with open(filename, 'w', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writeheader()
                csv_writer.writerow(data_dict)
    except IOError as e:
        print(f"Error: Could not write to file {filename}. {e}")
        success = False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        success = False
    return success

if __name__ == "__main__":
    create_static_csv("run_id_1", "../config/ppo/FoodCollector.yaml")
