#!/usr/bin/env python3
"""
Stage 11C: Validate reliable RFID identity anchors.

This stage characterises the reliable RFID anchors before identity
propagation. It evaluates:

1. consistency of RFID identities within each SLEAP track fragment;
2. distribution of anchors across animals;
3. distribution of anchors across RFID readers;
4. temporal coverage of anchors;
5. number of anchors supporting each track fragment.

Inputs
------
<recording>_rfid_reliable_anchors.csv

Outputs
-------
<recording>_rfid_anchor_track_summary.csv
<recording>_rfid_anchor_mouse_summary.csv
<recording>_rfid_anchor_reader_summary.csv
<recording>_rfid_anchor_time_summary.csv
<recording>_rfid_anchor_validation_summary.csv

Figures
-------
<recording>_rfid_anchor_timeline.png
<recording>_rfid_anchors_per_track.png
<recording>_rfid_anchors_per_reader.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "outputs/development_v1/identity/"
    "407405spareat1313_rfid_reliable_anchors.csv"
)

DEFAULT_OUTPUT_DIR = Path("outputs/development_v1/identity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate reliable RFID–SLEAP identity anchors."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Reliable anchor CSV. Default: {DEFAULT_INPUT}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--recording-id",
        type=str,
        default=None,
        help="Optional recording identifier for output filenames.",
    )

    parser.add_argument(
        "--time-bins",
        type=int,
        default=10,
        help="Number of equal-width temporal bins. Default: 10",
    )

    return parser.parse_args()


def infer_recording_id(input_path: Path) -> str:
    suffix = "_rfid_reliable_anchors"
    stem = input_path.stem

    if stem.endswith(suffix):
        return stem[: -len(suffix)]

    return stem


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def natural_track_key(track_id: object) -> tuple[str, int]:
    """
    Sort track labels naturally.

    For example:
    track_2 appears before track_10.
    """
    text = str(track_id)
    match = re.search(r"(\d+)$", text)

    if match:
        prefix = text[: match.start()]
        number = int(match.group(1))
        return prefix, number

    return text, -1


def choose_frame_column(df: pd.DataFrame) -> str:
    candidates = [
        "nearest_frame_idx",
        "frame_idx",
        "video_frame_idx",
        "matched_frame_idx",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify a frame-index column. Checked: "
        + ", ".join(candidates)
    )


def choose_time_column(df: pd.DataFrame) -> str | None:
    """
    Return a video-time column in seconds when one is available.
    """
    candidates = [
        "nearest_frame_time_s",
        "nearest_video_time_s",
        "video_time_s",
        "matched_video_time_s",
        "frame_time_s",
        "elapsed_video_time_s",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def safe_mode(series: pd.Series) -> object:
    modes = series.dropna().mode()

    if modes.empty:
        return np.nan

    return modes.iloc[0]


def create_track_summary(
    df: pd.DataFrame,
    frame_column: str,
    time_column: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for track_id, group in df.groupby(
        "assigned_track_id",
        dropna=False,
        sort=False,
    ):
        mouse_counts = (
            group["mouse_id"]
            .astype(str)
            .value_counts(dropna=False)
        )

        n_anchors = len(group)
        n_mouse_ids = int(mouse_counts.size)

        majority_mouse = safe_mode(group["mouse_id"])
        majority_count = int(mouse_counts.iloc[0]) if n_anchors else 0
        majority_fraction = (
            majority_count / n_anchors if n_anchors else np.nan
        )

        identities = "|".join(mouse_counts.index.astype(str).tolist())

        row = {
            "assigned_track_id": track_id,
            "n_anchors": n_anchors,
            "n_unique_mouse_ids": n_mouse_ids,
            "mouse_ids_observed": identities,
            "majority_mouse_id": majority_mouse,
            "majority_anchor_count": majority_count,
            "majority_identity_fraction": majority_fraction,
            "identity_conflict": n_mouse_ids > 1,
            "first_anchor_frame": pd.to_numeric(
                group[frame_column],
                errors="coerce",
            ).min(),
            "last_anchor_frame": pd.to_numeric(
                group[frame_column],
                errors="coerce",
            ).max(),
            "frame_span": (
                pd.to_numeric(group[frame_column], errors="coerce").max()
                - pd.to_numeric(group[frame_column], errors="coerce").min()
            ),
            "median_nearest_distance_cm": pd.to_numeric(
                group["nearest_distance_cm"],
                errors="coerce",
            ).median(),
            "median_candidate_margin_cm": pd.to_numeric(
                group["nearest_second_margin_cm"],
                errors="coerce",
            ).median(),
            "median_torso_score": pd.to_numeric(
                group["assigned_torso_score"],
                errors="coerce",
            ).median(),
        }

        if time_column is not None:
            times = pd.to_numeric(
                group[time_column],
                errors="coerce",
            )

            row["first_anchor_time_s"] = times.min()
            row["last_anchor_time_s"] = times.max()
            row["anchor_time_span_s"] = times.max() - times.min()

        rows.append(row)

    summary = pd.DataFrame(rows)

    ordered_tracks = sorted(
        summary["assigned_track_id"].tolist(),
        key=natural_track_key,
    )

    order_lookup = {
        track_id: index
        for index, track_id in enumerate(ordered_tracks)
    }

    summary["_sort_order"] = (
        summary["assigned_track_id"]
        .map(order_lookup)
    )

    summary = (
        summary
        .sort_values("_sort_order")
        .drop(columns="_sort_order")
        .reset_index(drop=True)
    )

    return summary


def create_mouse_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    summary = (
        df.groupby("mouse_id", dropna=False)
        .agg(
            n_anchors=("rfid_event_index", "size"),
            n_tracks=("assigned_track_id", "nunique"),
            n_readers=("reader", "nunique"),
            first_anchor_frame=("nearest_frame_idx", "min"),
            last_anchor_frame=("nearest_frame_idx", "max"),
            median_nearest_distance_cm=(
                "nearest_distance_cm",
                "median",
            ),
            median_candidate_margin_cm=(
                "nearest_second_margin_cm",
                "median",
            ),
            median_torso_score=(
                "assigned_torso_score",
                "median",
            ),
        )
        .reset_index()
    )

    summary["percentage_of_all_anchors"] = (
        100 * summary["n_anchors"] / total
        if total
        else np.nan
    )

    return summary.sort_values("mouse_id").reset_index(drop=True)


def create_reader_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    summary = (
        df.groupby("reader", dropna=False)
        .agg(
            n_anchors=("rfid_event_index", "size"),
            n_mice=("mouse_id", "nunique"),
            n_tracks=("assigned_track_id", "nunique"),
            median_nearest_distance_cm=(
                "nearest_distance_cm",
                "median",
            ),
            mean_nearest_distance_cm=(
                "nearest_distance_cm",
                "mean",
            ),
            median_candidate_margin_cm=(
                "nearest_second_margin_cm",
                "median",
            ),
        )
        .reset_index()
    )

    summary["percentage_of_all_anchors"] = (
        100 * summary["n_anchors"] / total
        if total
        else np.nan
    )

    return summary.sort_values("reader").reset_index(drop=True)


def create_time_summary(
    df: pd.DataFrame,
    frame_column: str,
    time_column: str | None,
    n_bins: int,
) -> tuple[pd.DataFrame, str, str]:
    """
    Divide the recording into equal-width temporal bins.

    Returns
    -------
    summary
        One row per temporal bin.
    plotting_column
        Column used for timeline plotting.
    plotting_label
        Human-readable x-axis label.
    """
    if n_bins < 1:
        raise ValueError("--time-bins must be at least 1.")

    if time_column is not None:
        plotting_column = time_column
        plotting_label = "Video time (minutes)"

        values = pd.to_numeric(
            df[time_column],
            errors="coerce",
        )

        values_for_bins = values
        units = "seconds"

    else:
        plotting_column = frame_column
        plotting_label = "Video frame"

        values = pd.to_numeric(
            df[frame_column],
            errors="coerce",
        )

        values_for_bins = values
        units = "frames"

    valid = values_for_bins.dropna()

    if valid.empty:
        raise ValueError(
            "No valid temporal values were available for time analysis."
        )

    minimum = float(valid.min())
    maximum = float(valid.max())

    if minimum == maximum:
        edges = np.array([minimum, maximum + 1])
    else:
        edges = np.linspace(
            minimum,
            maximum,
            n_bins + 1,
        )

    working = df.copy()

    working["_temporal_value"] = values_for_bins
    working["_time_bin"] = pd.cut(
        working["_temporal_value"],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    summary = (
        working.groupby(
            "_time_bin",
            observed=False,
            dropna=False,
        )
        .agg(
            n_anchors=("rfid_event_index", "size"),
            n_mice=("mouse_id", "nunique"),
            n_readers=("reader", "nunique"),
            n_tracks=("assigned_track_id", "nunique"),
        )
        .reset_index()
    )

    summary = summary[
        summary["_time_bin"].notna()
    ].copy()

    summary["bin_number"] = np.arange(1, len(summary) + 1)
    summary["bin_start"] = [
        float(interval.left)
        for interval in summary["_time_bin"]
    ]

    summary["bin_end"] = [
        float(interval.right)
        for interval in summary["_time_bin"]
    ]

    summary["bin_midpoint"] = (
        summary["bin_start"].astype(float)
        + summary["bin_end"].astype(float)
    ) / 2.0

    summary["temporal_units"] = units

    summary = summary[
        [
            "bin_number",
            "bin_start",
            "bin_end",
            "bin_midpoint",
            "temporal_units",
            "n_anchors",
            "n_mice",
            "n_readers",
            "n_tracks",
        ]
    ]

    return summary, plotting_column, plotting_label


def save_anchor_timeline(
    df: pd.DataFrame,
    plotting_column: str,
    plotting_label: str,
    output_path: Path,
) -> None:
    plot_df = df.copy()

    x = pd.to_numeric(        plot_df[plotting_column],
        errors="coerce",
    )

    if "time" in plotting_column.lower():
        x = x / 60

    mouse_labels = sorted(
        plot_df["mouse_id"].dropna().astype(str).unique()
    )

    mouse_positions = {
        mouse: index
        for index, mouse in enumerate(mouse_labels)
    }

    y = (
        plot_df["mouse_id"]
        .astype(str)
        .map(mouse_positions)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.scatter(
        x,
        y,
        s=24,
        alpha=0.75,
    )

    ax.set_yticks(range(len(mouse_labels)))
    ax.set_yticklabels(mouse_labels)
    ax.set_xlabel(plotting_label)
    ax.set_ylabel("RFID mouse identity")
    ax.set_title("Reliable RFID identity anchors across the recording")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_anchors_per_track(
    track_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = track_summary.copy()

    fig_width = max(9, len(plot_df) * 0.32)

    fig, ax = plt.subplots(
        figsize=(fig_width, 5.5)
    )

    positions = np.arange(len(plot_df))

    ax.bar(
        positions,
        plot_df["n_anchors"],
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        plot_df["assigned_track_id"],
        rotation=90,
    )

    ax.set_xlabel("SLEAP track fragment")
    ax.set_ylabel("Reliable RFID anchors")
    ax.set_title("Reliable RFID anchors supporting each SLEAP track")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_anchors_per_reader(
    reader_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        reader_summary["reader"].astype(str),
        reader_summary["n_anchors"],
    )

    ax.set_xlabel("RFID reader")
    ax.set_ylabel("Reliable RFID anchors")
    ax.set_title("Reliable identity anchors contributed by each reader")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    recording_id = (
        args.recording_id
        or infer_recording_id(input_path)
    )

    df = pd.read_csv(input_path)

    required_columns = [
        "rfid_event_index",
        "mouse_id",
        "reader",
        "assigned_track_id",
        "nearest_distance_cm",
        "nearest_second_margin_cm",
        "assigned_torso_score",
    ]
    require_columns(df, required_columns)

    frame_column = choose_frame_column(df)
    time_column = choose_time_column(df)

    # Standardise the frame-column name for downstream summary functions.
    if frame_column != "nearest_frame_idx":
        df["nearest_frame_idx"] = df[frame_column]
        frame_column = "nearest_frame_idx"

    track_summary = create_track_summary(
        df=df,
        frame_column=frame_column,
        time_column=time_column,
    )

    mouse_summary = create_mouse_summary(df)
    reader_summary = create_reader_summary(df)

    (
        time_summary,
        plotting_column,
        plotting_label,
    ) = create_time_summary(
        df=df,
        frame_column=frame_column,
        time_column=time_column,
        n_bins=args.time_bins,
    )

    total_anchors = len(df)
    total_tracks = df["assigned_track_id"].nunique(dropna=True)
    anchored_mice = df["mouse_id"].nunique(dropna=True)
    active_readers = df["reader"].nunique(dropna=True)

    conflicting_tracks = int(
        track_summary["identity_conflict"].sum()
    )

    consistent_tracks = int(
        (~track_summary["identity_conflict"]).sum()
    )

    single_anchor_tracks = int(
        (track_summary["n_anchors"] == 1).sum()
    )

    multi_anchor_tracks = int(
        (track_summary["n_anchors"] > 1).sum()
    )

    minimum_anchors_per_track = int(
        track_summary["n_anchors"].min()
    )

    median_anchors_per_track = float(
        track_summary["n_anchors"].median()
    )

    maximum_anchors_per_track = int(
        track_summary["n_anchors"].max()
    )

    bins_with_anchors = int(
        (time_summary["n_anchors"] > 0).sum()
    )

    total_bins = len(time_summary)

    validation_summary = pd.DataFrame(
        [
            {
                "recording_id": recording_id,
                "total_reliable_anchors": total_anchors,
                "anchored_track_fragments": total_tracks,
                "anchored_mouse_ids": anchored_mice,
                "active_readers": active_readers,
                "identity_consistent_tracks": consistent_tracks,
                "identity_conflicting_tracks": conflicting_tracks,
                "identity_conflict_rate_percent": (
                    100 * conflicting_tracks / total_tracks
                    if total_tracks
                    else np.nan
                ),
                "single_anchor_tracks": single_anchor_tracks,
                "multi_anchor_tracks": multi_anchor_tracks,
                "minimum_anchors_per_track": minimum_anchors_per_track,
                "median_anchors_per_track": median_anchors_per_track,
                "maximum_anchors_per_track": maximum_anchors_per_track,
                "temporal_bins": total_bins,
                "temporal_bins_with_anchors": bins_with_anchors,
                "temporal_coverage_percent": (
                    100 * bins_with_anchors / total_bins
                    if total_bins
                    else np.nan
                ),
                "first_anchor_frame": pd.to_numeric(
                    df[frame_column],
                    errors="coerce",
                ).min(),
                "last_anchor_frame": pd.to_numeric(
                    df[frame_column],
                    errors="coerce",
                ).max(),
            }
        ]
    )

    if time_column is not None:
        validation_summary["first_anchor_time_s"] = (
            pd.to_numeric(
                df[time_column],
                errors="coerce",
            ).min()
        )

        validation_summary["last_anchor_time_s"] = (
            pd.to_numeric(
                df[time_column],
                errors="coerce",
            ).max()
        )

    track_path = (
        output_dir
        / f"{recording_id}_rfid_anchor_track_summary.csv"
    )

    mouse_path = (
        output_dir
        / f"{recording_id}_rfid_anchor_mouse_summary.csv"
    )

    reader_path = (
        output_dir
        / f"{recording_id}_rfid_anchor_reader_summary.csv"
    )

    time_path = (
        output_dir
        / f"{recording_id}_rfid_anchor_time_summary.csv"
    )

    validation_path = (
        output_dir
        / f"{recording_id}_rfid_anchor_validation_summary.csv"
    )

    track_summary.to_csv(track_path, index=False)
    mouse_summary.to_csv(mouse_path, index=False)
    reader_summary.to_csv(reader_path, index=False)
    time_summary.to_csv(time_path, index=False)
    validation_summary.to_csv(validation_path, index=False)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = (
        figure_dir
        / f"{recording_id}_rfid_anchor_timeline.png"
    )

    track_figure_path = (
        figure_dir
        / f"{recording_id}_rfid_anchors_per_track.png"
    )

    reader_figure_path = (
        figure_dir
        / f"{recording_id}_rfid_anchors_per_reader.png"
    )

    save_anchor_timeline(
        df=df,
        plotting_column=plotting_column,
        plotting_label=plotting_label,
        output_path=timeline_path,
    )

    save_anchors_per_track(
        track_summary=track_summary,
        output_path=track_figure_path,
    )

    save_anchors_per_reader(
        reader_summary=reader_summary,
        output_path=reader_figure_path,
    )

    print()
    print("Stage 11C: RFID anchor validation")
    print("=" * 43)
    print(f"Input:                     {input_path}")
    print(f"Recording:                 {recording_id}")
    print()
    print(f"Reliable anchors:          {total_anchors}")
    print(f"Anchored track fragments:  {total_tracks}")
    print(f"RFID mouse identities:     {anchored_mice}")
    print(f"Readers represented:       {active_readers}")
    print()
    print(f"Consistent tracks:         {consistent_tracks}")
    print(f"Conflicting tracks:        {conflicting_tracks}")
    print(
        "Conflict rate:             "
        f"{100 * conflicting_tracks / total_tracks:.2f}%"
        if total_tracks
        else "Conflict rate:             NA"
    )
    print()
    print(f"Single-anchor tracks:      {single_anchor_tracks}")
    print(f"Multi-anchor tracks:       {multi_anchor_tracks}")
    print(
        f"Anchors per track:         "
        f"min={minimum_anchors_per_track}, "
        f"median={median_anchors_per_track:.1f}, "
        f"max={maximum_anchors_per_track}"
    )
    print()
    print(
        f"Temporal bins covered:     "
        f"{bins_with_anchors}/{total_bins} "
        f"({100 * bins_with_anchors / total_bins:.1f}%)"
    )
    print()
    print("Anchors by mouse:")
    print(
        mouse_summary[
            [
                "mouse_id",
                "n_anchors",
                "percentage_of_all_anchors",
                "n_tracks",
            ]
        ].to_string(index=False)
    )
    print()
    print("Anchors by reader:")
    print(
        reader_summary[
            [
                "reader",
                "n_anchors",
                "percentage_of_all_anchors",
                "n_tracks",
            ]
        ].to_string(index=False)
    )
    print()
    print("Saved:")
    print(f"  Track summary:           {track_path}")
    print(f"  Mouse summary:           {mouse_path}")
    print(f"  Reader summary:          {reader_path}")
    print(f"  Time summary:            {time_path}")
    print(f"  Validation summary:      {validation_path}")
    print(f"  Timeline figure:         {timeline_path}")
    print(f"  Track figure:            {track_figure_path}")
    print(f"  Reader figure:           {reader_figure_path}")
    print()


if __name__ == "__main__":
    main()
