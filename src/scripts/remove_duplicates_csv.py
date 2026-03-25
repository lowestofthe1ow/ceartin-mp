import argparse
import sys

import pandas as pd


def clean_csv(file_path):
    try:
        df = pd.read_csv(file_path)

        original_count = len(df)

        df_cleaned = df.dropna()

        removed_count = original_count - len(df_cleaned)

        df_cleaned.to_csv(file_path, index=False)

        print(f"Rows removed: {removed_count}")
        print(f"Final row count: {len(df_cleaned)}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Remove rows with blank entries from a CSV file and overwrite it."
    )

    parser.add_argument("path", help="Path to the CSV file you want to clean")

    args = parser.parse_args()
    clean_csv(args.path)


if __name__ == "__main__":
    main()
