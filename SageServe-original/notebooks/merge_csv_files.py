import os
import pandas as pd

def merge_csv_files(folder_path, sorting_column, output_file='merged_output.csv'):
    """
    Merge all CSV files in a specified folder into a single CSV file.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing CSV files to merge
    output_file : str, optional
        Name of the output merged CSV file (default is 'merged_output.csv')
    
    Returns:
    --------
    str
        Path to the merged CSV file
    
    Raises:
    -------
    ValueError
        If no CSV files are found in the specified folder
    """
    # Get a list of all CSV files in the folder, sorted alphabetically
    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    
    # Check if any CSV files were found
    if not csv_files:
        raise ValueError(f"No CSV files found in the folder: {folder_path}")
    
    # List to store DataFrames
    dataframes = []
    
    # Read each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add a column to track the source file if desired
            df['source_file'] = csv_file
            
            dataframes.append(df)
            print(f"Processed file: {csv_file}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Construct the full output path
    output_path = os.path.join(folder_path, output_file)
    
    merged_df = merged_df.sort_values(by=sorting_column)

    # Save the merged DataFrame
    merged_df.to_csv(output_path, index=False)
    
    print(f"Merged {len(csv_files)} CSV files into {output_file}")
    
    return output_path

def get_output_dir(seed=0, trace_filename='ES_26', feed_async=True, feed_async_granularity=1,
                   scaling_level=2, scaling_interval=-1, lts=False, sts=True):
    output_dir = f'../results/{seed}/{trace_filename}/feed_async_{feed_async}/feed_async_granularity_{feed_async_granularity}/scaling_level_{scaling_level}/scaling_interval_{scaling_interval}/lts_{lts}_sts{sts}'
    assert os.path.isdir(output_dir), f"Directory {output_dir} does not exist"
    return output_dir

# Example usage
if __name__ == "__main__":
    try:
        # Replace 'path/to/your/csv/folder' with the actual path to your folder
        merged_file = merge_csv_files(f"{get_output_dir()}/memory/", "time")
        merged_file = merge_csv_files(f"{get_output_dir()}/region_routers/centralus", "arrival_timestamp")
        merged_file = merge_csv_files(f"{get_output_dir()}/region_routers/westus", "arrival_timestamp")
        merged_file = merge_csv_files(f"{get_output_dir()}/region_routers/eastus", "arrival_timestamp")
        print(f"Merged CSV saved to: {merged_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
