import csv
import os
import glob

def process_files(directory, reward_type):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(directory, reward_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Pattern to match specific reward type files
    file_pattern = f"{directory}/{reward_type}_*.csv"
    files = glob.glob(file_pattern)

    for file_path in files:
        method = file_path.split('_')[-1].split('.')[0]  # Extract method from filename
        a_rows, b_rows = [], []
        
        with open(file_path, 'r', newline='') as infile:
            reader = csv.reader(infile)
            # Skip the first two lines
            for _ in range(2):
                next(reader, None)

            # Process lines from the third to the thirty-second
            lines_to_process = 30  # number of lines to process after the first two
            for index, row in enumerate(reader):
                if index >= lines_to_process:
                    break
                if index % 3 == 0:  # A class data
                    a_rows.append(row)
                elif index % 3 == 2:  # B class data, skip index 1 as it's an empty line
                    b_rows.append(row)

        # Save A class numbers
        output_path_a = os.path.join(output_dir, f"{method}.csv")
        with open(output_path_a, 'w', newline='') as outfile_a:
            writer = csv.writer(outfile_a)
            for row in a_rows:
                writer.writerow(row)

        # Save B class numbers
        output_path_b = os.path.join(output_dir, f"{method}_donuts.csv")
        with open(output_path_b, 'w', newline='') as outfile_b:
            writer = csv.writer(outfile_b)
            for row in b_rows:
                writer.writerow(row)

# Example usage
base_directory = 'datasets/donut-dqn'  # Base directory containing the CSV files
input_reward_type = "utilitarian"  # Get reward type from user
process_files(base_directory, input_reward_type)
