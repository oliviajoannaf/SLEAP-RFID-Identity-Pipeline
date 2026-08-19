#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


def main():

    parser = argparse.ArgumentParser(
        description="Merge SLEAP coordinate table with Basler timestamps."
    )

    parser.add_argument("sleap_csv", type=Path)
    parser.add_argument("timestamp_csv", type=Path)
    parser.add_argument("output_csv", type=Path)

    args = parser.parse_args()

    print("Loading SLEAP coordinate table...")
    sleap = pd.read_csv(args.sleap_csv)

    print("Loading Basler timestamp table...")
    timestamps = pd.read_csv(
        args.timestamp_csv,
        sep=";",
        header=None,
        names=[
            "frame_idx",
            "absolute_timestamp",
            "elapsed_seconds",
        ],
    )

    print("Merging tables...")

    merged = sleap.merge(
        timestamps,
        on="frame_idx",
        how="left",
        validate="many_to_one",
    )

    missing = merged["absolute_timestamp"].isna().sum()

    if missing != 0:
        raise RuntimeError(
            f"{missing} rows have no matching timestamp."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    print("\nAlignment complete")
    print(f"SLEAP rows:      {len(sleap)}")
    print(f"Timestamp rows:  {len(timestamps)}")
    print(f"Merged rows:     {len(merged)}")
    print(f"Missing matches: {missing}")

    print("\nOutput:")
    print(args.output_csv)


if __name__ == "__main__":
    main()
