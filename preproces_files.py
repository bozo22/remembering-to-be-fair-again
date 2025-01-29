import pandas as pd

def split_csv_by_data_type(input_file, integers_file, floats_file):
    """
    Splits a CSV file into two files: one containing rows with integers and the other with floats.
    
    Args:
    - input_file (str): Path to the input CSV file.
    - integers_file (str): Path to save the file containing rows of integers.
    - floats_file (str): Path to save the file containing rows of floats.
    
    Returns:
    - None: The function writes the output files directly.
    """
    integers_list = []
    floats_list = []

    with open(input_file, 'r') as file:
        for line in file:
            try:
                # Attempt to convert the line to integers
                integers = list(map(int, line.strip().split(',')))
                integers_list.append(integers)
            except ValueError:
                try:
                    # If integers fail, attempt to convert the line to floats
                    floats = list(map(float, line.strip().split(',')))
                    floats_list.append(floats)
                except ValueError:
                    # Skip lines that cannot be converted to either type
                    pass

    # Save the results to CSV files
    pd.DataFrame(integers_list).to_csv(integers_file, index=False, header=False)
    pd.DataFrame(floats_list).to_csv(floats_file, index=False, header=False)

# Example usage:
split_csv_by_data_type('datasets/donut/binary-desNO-MEMORY-people5-cfFalse-rtnsw-20250128_185736.csv', 'datasets/donut/NoMemorydonuts.csv', 'datasets/donut/NoMemoryprob.csv')
