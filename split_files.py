import pandas as pd
import os

def split_csv_into_two(file_path, output_dir):
    """
    Split a CSV file into two smaller files of approximately equal size.

    Parameters:
        file_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the smaller files.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path).split('.')[0]

    # Calculate the split point
    split_index = len(df) // 2

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the first half
    first_half = df.iloc[:split_index]
    first_half_file = os.path.join(output_dir, f"{file_name}_part1.csv")
    first_half.to_csv(first_half_file, index=False)
    print(f"Saved {first_half_file}")

    # Save the second half
    second_half = df.iloc[split_index:]
    second_half_file = os.path.join(output_dir, f"{file_name}_part2.csv")
    second_half.to_csv(second_half_file, index=False)
    print(f"Saved {second_half_file}")

# Example usage: Splitting two CSV files
split_csv_into_two('clean_df_3_single_weapon.csv', 'output_chunks')
split_csv_into_two('crime_dumscalab.csv', 'output_chunks')
