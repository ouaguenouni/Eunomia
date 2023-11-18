import os
import datetime
import json
import yaml
import glob
import re

def load_yaml_files_from_directory(directory):
    """
    Load all YAML files from a specified directory and return a list of dictionaries.
    Handles Python-specific tags using FullLoader.

    Parameters:
    directory (str): The path to the directory containing the YAML files.

    Returns:
    list: A list of dictionaries extracted from the YAML files.
    """
    yaml_files = glob.glob(f"{directory}/*.yaml")
    all_dicts = []

    for file_path in yaml_files:
        with open(file_path, 'r') as file:
            try:
                data = yaml.load(file, Loader=yaml.FullLoader)
                if isinstance(data, dict):
                    all_dicts.append(data)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file {file_path}: {exc}")

    return all_dicts


def compute_experiment_file_name(experiment_params, experiment_name):
    """
    Computes the file name for an experiment based on the experiment parameters, name, and the current date.
    The file name is formatted as 'experiment_name/YYYY-MM-DD/arg1-value1_arg2-value2...argN-valueN.yaml'.

    Parameters:
    experiment_params (dict): The parameters of the experiment. Each key-value pair in the dictionary
                              represents a parameter and its value.
    experiment_name (str): The name of the experiment, which is used as the prefix in the file path.

    Returns:
    str: The full path for the file within the experiment's directory structure.
    """
    # Format the current date for the directory name
    now = datetime.datetime.now()
    date_dir = now.strftime("%Y-%m-%d")

    # Format experiment parameters for the file name
    params_str = "_".join([f"{key}-{value}" for key, value in experiment_params.items()])

    # Create the full file path
    file_path = os.path.join(experiment_name, date_dir, f"{params_str}.yaml")

    return file_path

def record_experiment_results(experiment_params, experiment_name):
    """
    Records the results of an experiment by creating the necessary directories and a file.
    The directory structure is 'experiment_name/YYYY-MM-DD/'.

    Parameters:
    experiment_params (dict): The parameters of the experiment, used in the file name.
    experiment_name (str): The name of the experiment, used for the main directory name.

    Returns:
    str: The path to the created file.
    """
    # Compute the full file path
    file_path = compute_experiment_file_name(experiment_params, experiment_name)

    # Create the necessary directories
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create an empty file with the generated name
    with open(file_path, 'w') as file:
        pass

    return file_path



def find_experiment_file(experiment_params, experiment_name):
    """
    Find a file that was created for a specific experiment with given parameters,
    across all dates.

    Parameters:
    experiment_params (dict): The parameters of the experiment used in the file name.
    experiment_name (str): The name of the experiment used for the main directory name.

    Returns:
    str: The path to the found file, or None if no file is found.
    """
    # Compute expected file name without date
    expected_file_name = compute_experiment_file_name(experiment_params, experiment_name).split('/')[-1]

    # Search for the file across all dates
    search_pattern = os.path.join(experiment_name, "*", expected_file_name)
    found_files = glob.glob(search_pattern)

    if found_files:
        return found_files[0]

    return None