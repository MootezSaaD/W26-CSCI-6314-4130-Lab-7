import json
import os
import csv

def read_jsonl(file_path):
    """
    Reads a .jsonl file and returns the data as a list of dictionaries.

    Args:
        file_path (str): Path to the .jsonl file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a JSON object from the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object and append to the list
            data.append(json.loads(line.strip()))
    return data

def read_json(file_path):
    """
    Reads a .json file and returns the data as a Python object.

    Args:
        file_path (str): Path to the .json file.

    Returns:
        dict or list: The data loaded from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load JSON data from the file
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}. Error: {e}", e.doc, e.pos)

def process_json(file_path, output_dir, csv_file_path):
    """
    Reads a .jsonl file, extracts the "func" and "target" fields, writes the content of "func"
    to a file named <ID>_<LABEL>.txt, and generates a CSV file mapping file paths to labels.

    Args:
        file_path (str): Path to the .jsonl file.
        output_dir (str): Directory to save the output files.
        csv_file_path (str): Path to save the CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List to store CSV rows
    csv_rows = []
    ID = 0
    samples = read_json(file_path)

    for line in samples:
        # Parse the JSON object
        data = line
        # Extract the required fields
        func_content = data.get("func", "")
        target_label = data.get("target", "")
        
        # Create the output file name
        output_file_name = f"func_{ID}_{target_label}.txt"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        # Write the "func" content to the file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(func_content)
        
        # Add the file path and label to the CSV rows
        csv_rows.append([output_file_path, target_label])
        
        print(f"Created file: {output_file_path}")

        ID += 1


    # Write the CSV file
    with open(csv_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_path", "label"])  # Write header
        writer.writerows(csv_rows)  # Write rows

    print(f"CSV file created: {csv_file_path}")