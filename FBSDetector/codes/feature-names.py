import os
import pandas as pd

def print_column_names(file_path, file_name, output_file):
    try:
        df = pd.read_csv(file_path)
        with open(output_file, 'a') as f:
            f.write(f"\nColumns in {file_name}:\n")
            f.write("-" * (len(f"Columns in {file_name}:")) + "\n")
            for col in df.columns:
                f.write(f"- {col}\n")
    except FileNotFoundError:
        with open(output_file, 'a') as f:
            f.write(f"Error: {file_name} not found at {file_path}\n")
    except Exception as e:
        with open(output_file, 'a') as f:
            f.write(f"Error reading {file_name}: {str(e)}\n")

def main():
    # File paths
    fbs_file = "dataset/fbs_nas.csv"
    msa_file = "dataset/fbs_rrc.csv"
    # check if the output directory exists else create it
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = "outputs/column_names_output.txt"

    # Clear the output file before writing
    open(output_file, 'w').close()

    # Print column names for both files
    print_column_names(fbs_file, "fbs_nas.csv", output_file)
    print_column_names(msa_file, "fbs_rrc.csv", output_file)

if __name__ == "__main__":
    main()