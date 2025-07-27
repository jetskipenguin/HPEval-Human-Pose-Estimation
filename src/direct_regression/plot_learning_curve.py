import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os

def create_plot(file_path, output_path):
    """
    Reads data from a single-column CSV file, generates a plot,
    and saves it to a file. The y-axis represents the values from
    the CSV, and the x-axis represents the row number.

    Args:
        file_path (str): The path to the input CSV file.
        output_path (str): The path where the plot image will be saved.
    """
    try:
        df = pd.read_csv(file_path, header=None, names=['value'])
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['value'], marker='o', linestyle='-', markersize=4)

    plt.title(f'Average Precision (AP) per Epoch for {file_path.split(os.sep)[-1]}', fontsize=16)
    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('AP', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print(f"Plot successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error: Could not save the plot to '{output_path}'. Reason: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a plot from a single-column CSV file and save it."
    )

    parser.add_argument(
        '--file_path',
        type=str,
        help='The path to the CSV file to be plotted.'
    )

    args = parser.parse_args()

    base_path, _ = os.path.splitext(args.file_path)
    output_file_path = base_path + '-learning-curve-plot.png'

    create_plot(args.file_path, output_file_path)
