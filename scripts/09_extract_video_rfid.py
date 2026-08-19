#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


REQUIRED_SLEAP_COLUMNS = {
    "frame_idx",
    "absolute_timestamp",
    "elapsed_seconds",
}

REQUIRED_RFID_COLUMNS = {
    "timestamp",
    "mouse_id",
    "reader",
    "duration_ms",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract RFID events overlapping one video and align each event "
            "to the nearest timestamped SLEAP frame."
        )
    )

    parser.add_argument(
        "sleap_csv",
        type=Path,
        help="Timestamped SLEAP coordinate CSV produced by Stage 06.",
    )
    parser.add_argument(
        "rfid_csv",
        type=Path,
        help="Mapped RFID event CSV produced by Stage 08.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Output CSV containing video-specific, frame-aligned RFID events.",
    )
    parser.add_argument(
        "summary_txt",
        type=Path,
        help="Output text file containing alignment and validation statistics.",
    )

    return parser.parse_args()


def check_columns(df, required, table_name):
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}"
        )


def main():
    args = parse_args()

    print("Loading timestamped SLEAP table...")
    sleap = pd.read_csv(args.sleap_csv)

    print("Loading mapped RFID events...")
    rfid = pd.read_csv(args.rfid_csv)

    check_columns(sleap, REQUIRED_SLEAP_COLUMNS, "SLEAP table")
    check_columns(rfid, REQUIRED_RFID_COLUMNS, "RFID table")

    # Parse timestamps.
    sleap["absolute_timestamp"] = pd.to_datetime(
        sleap["absolute_timestamp"],
        errors="raise",
    )

    rfid["event_start_timestamp"] = pd.to_datetime(
        rfid["timestamp"],
        errors="raise",
    )

    # Each SLEAP frame can contain multiple animal instances.
    # Retain one timestamp record per frame.
    frames = (
        sleap[
            ["frame_idx", "absolute_timestamp", "elapsed_seconds"]
        ]
        .drop_duplicates(subset="frame_idx")
        .sort_values("absolute_timestamp")
        .reset_index(drop=True)
    )

    if frames.empty:
        raise ValueError("The SLEAP table contains no frame timestamps.")

    if frames["frame_idx"].duplicated().any():
        raise RuntimeError("Duplicate frame indices remain after processing.")

    video_start = frames["absolute_timestamp"].min()
    video_end = frames["absolute_timestamp"].max()
    video_duration = (
        frames["elapsed_seconds"].max()
        - frames["elapsed_seconds"].min()
    )

    # RFID event intervals.
    rfid["duration_ms"] = pd.to_numeric(
        rfid["duration_ms"],
        errors="raise",
    )

    if (rfid["duration_ms"] < 0).any():
        raise ValueError("Negative RFID event durations were detected.")

    rfid["event_end_timestamp"] = (
        rfid["event_start_timestamp"]
        + pd.to_timedelta(rfid["duration_ms"], unit="ms")
    )

    # Keep events whose time interval overlaps the video interval.
    overlap_mask = (
        (rfid["event_end_timestamp"] >= video_start)
        & (rfid["event_start_timestamp"] <= video_end)
    )

    selected = rfid.loc[overlap_mask].copy()

    if selected.empty:
        raise RuntimeError(
            "No RFID events overlap the SLEAP video interval. "
            "Check the recording dates, clocks and input files."
        )

    # Restrict each event to the part that actually overlaps the video.
    selected["overlap_start_timestamp"] = (
        selected["event_start_timestamp"].clip(lower=video_start)
    )
    selected["overlap_end_timestamp"] = (
        selected["event_end_timestamp"].clip(upper=video_end)
    )

    selected["overlap_duration_ms"] = (
        selected["overlap_end_timestamp"]
        - selected["overlap_start_timestamp"]
    ).dt.total_seconds() * 1000.0

    # Use the midpoint of the overlapping interval as the representative
    # matching time. This is more robust than using only the first detection.
    selected["matching_timestamp"] = (
        selected["overlap_start_timestamp"]
        + (
            selected["overlap_end_timestamp"]
            - selected["overlap_start_timestamp"]
        ) / 2
    )

    selected = selected.sort_values("matching_timestamp").reset_index(
        drop=True
    )

    # Match each event to the nearest Basler/SLEAP frame timestamp.
    aligned = pd.merge_asof(
        selected,
        frames.rename(
            columns={
                "frame_idx": "nearest_frame_idx",
                "absolute_timestamp": "nearest_frame_timestamp",
                "elapsed_seconds": "video_elapsed_seconds",
            }
        ),
        left_on="matching_timestamp",
        right_on="nearest_frame_timestamp",
        direction="nearest",
    )

    aligned["frame_time_difference_ms"] = (
        aligned["matching_timestamp"]
        - aligned["nearest_frame_timestamp"]
    ).dt.total_seconds().abs() * 1000.0

    aligned.insert(
        0,
        "video_start_timestamp",
        video_start,
    )
    aligned.insert(
        1,
        "video_end_timestamp",
        video_end,
    )

    # Validation.
    if aligned["nearest_frame_idx"].isna().any():
        raise RuntimeError("Some RFID events were not assigned to a frame.")

    if len(aligned) != int(overlap_mask.sum()):
        raise RuntimeError(
            "RFID row count changed unexpectedly during frame alignment."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_txt.parent.mkdir(parents=True, exist_ok=True)

    aligned.to_csv(args.output_csv, index=False)

    reader_counts = (
        aligned["reader"]
        .value_counts()
        .sort_index()
    )

    mouse_counts = (
        aligned["mouse_id"]
        .value_counts()
        .sort_index()
    )

    summary_lines = [
        "Stage 09 – Video-specific RFID extraction and frame alignment",
        "",
        f"SLEAP input: {args.sleap_csv}",
        f"RFID input: {args.rfid_csv}",
        f"Output: {args.output_csv}",
        "",
        f"Video start: {video_start}",
        f"Video end: {video_end}",
        f"Video duration seconds: {video_duration:.6f}",
        f"Unique video frames: {len(frames)}",
        "",
        f"RFID input events: {len(rfid)}",
        f"RFID events overlapping video: {len(aligned)}",
        f"RFID events outside video: {len(rfid) - len(aligned)}",
        "",
        (
            "Median nearest-frame difference ms: "
            f"{aligned['frame_time_difference_ms'].median():.3f}"
        ),
        (
            "Maximum nearest-frame difference ms: "
            f"{aligned['frame_time_difference_ms'].max():.3f}"
        ),
        "",
        "Events by mouse:",
    ]

    for mouse_id, count in mouse_counts.items():
        summary_lines.append(f"  {mouse_id}: {count}")

    summary_lines.extend(["", "Events by reader:"])

    for reader, count in reader_counts.items():
        summary_lines.append(f"  {reader}: {count}")

    summary_lines.extend(
        [
            "",
            "Validation:",
            "  All retained RFID events overlap the video interval.",
            "  Every retained RFID event was assigned to a SLEAP frame.",
            "  Input RFID data were not modified or discarded except by "
            "video-interval filtering.",
        ]
    )

    args.summary_txt.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nStage 09 complete")
    print(f"Video start:              {video_start}")
    print(f"Video end:                {video_end}")
    print(f"Unique frames:            {len(frames)}")
    print(f"Input RFID events:        {len(rfid)}")
    print(f"Events inside video:      {len(aligned)}")
    print(f"Events outside video:     {len(rfid) - len(aligned)}")
    print(
        "Median frame difference: "
        f"{aligned['frame_time_difference_ms'].median():.3f} ms"
    )
    print(
        "Maximum frame difference: "
        f"{aligned['frame_time_difference_ms'].max():.3f} ms"
    )
    print(f"\nOutput CSV: {args.output_csv}")
    print(f"Summary:    {args.summary_txt}")


if __name__ == "__main__":
    main()
