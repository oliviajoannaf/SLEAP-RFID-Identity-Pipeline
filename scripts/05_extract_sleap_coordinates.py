#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io as sio


def extract_coordinates(input_slp: Path) -> pd.DataFrame:
    labels = sio.load_file(input_slp)

    rows: list[dict[str, object]] = []

    for labeled_frame in labels.labeled_frames:
        frame_idx = labeled_frame.frame_idx

        for instance_index, instance in enumerate(labeled_frame.instances):
            track_name = (
                instance.track.name
                if instance.track is not None
                else "untracked"
            )

            row: dict[str, object] = {
                "frame_idx": frame_idx,
                "instance_index": instance_index,
                "track_id": track_name,
                "instance_score": float(instance.score),
                "tracking_score": float(instance.tracking_score),
                "n_visible": int(instance.n_visible),
                "centroid_x": float(instance.centroid_xy[0]),
                "centroid_y": float(instance.centroid_xy[1]),
            }

            for point in instance.points:
                node_name = str(point["name"])
                xy = point["xy"]

                row[f"{node_name}_x"] = (
                    float(xy[0]) if not np.isnan(xy[0]) else np.nan
                )
                row[f"{node_name}_y"] = (
                    float(xy[1]) if not np.isnan(xy[1]) else np.nan
                )
                row[f"{node_name}_score"] = float(point["score"])
                row[f"{node_name}_visible"] = bool(point["visible"])

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frame-wise SLEAP predictions to CSV."
    )
    parser.add_argument(
        "input_slp",
        type=Path,
        help="Path to SLEAP prediction .slp file.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path for extracted CSV output.",
    )
    args = parser.parse_args()

    dataframe = extract_coordinates(args.input_slp)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output_csv, index=False)

    print(f"Input: {args.input_slp}")
    print(f"Output: {args.output_csv}")
    print(f"Rows written: {len(dataframe)}")
    print(f"Columns written: {len(dataframe.columns)}")
    print(f"Unique frames: {dataframe['frame_idx'].nunique()}")
    print(f"Unique SLEAP track IDs: {dataframe['track_id'].nunique()}")


if __name__ == "__main__":
    main()
