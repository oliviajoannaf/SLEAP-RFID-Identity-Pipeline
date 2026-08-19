#!/usr/bin/env python3
"""
Stage 11B: Filter provisional RFID–SLEAP assignments into reliable anchors.

The script removes assignments that were flagged during Stage 11A as:

1. spatially distant from the activated RFID reader; or
2. ambiguous because the nearest and second-nearest SLEAP candidates were
   insufficiently separated.

SLEAP confidence scores are retained as diagnostics but are not used as
exclusion criteria at this stage.

Outputs
-------
<recording>_rfid_reliable_anchors.csv
<recording>_rfid_rejected_anchors.csv
<recording>_rfid_anchor_filter_summary.csv
<recording>_rfid_anchor_filter_statistics.csv

Figures
-------
<recording>_anchor_distance_distribution.png
<recording>_anchor_margin_distribution.png
<recording>_anchor_torso_score_distribution.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "outputs/development_v1/identity/"
    "407405spareat1313_rfid_anchor_assignments.csv"
)

DEFAULT_OUTPUT_DIR = Path("outputs/development_v1/identity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter provisional RFID assignments into reliable anchors."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Stage 11A assignment CSV. Default: {DEFAULT_INPUT}",
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
        help=(
            "Optional recording identifier for output filenames. "
            "By default it is inferred from the input filename."
        ),
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def normalise_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """Convert bool, 0/1, or common string representations into Boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        invalid = ~series.fillna(0).isin([0, 1])
        if invalid.any():
            bad_values = sorted(series[invalid].dropna().unique().tolist())
            raise ValueError(
                f"Unexpected numeric values in {column_name}: {bad_values}"
            )
        return series.fillna(0).astype(int).astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    cleaned = series.astype(str).str.strip().str.lower()
    converted = cleaned.map(mapping)

    invalid = converted.isna() & series.notna()
    if invalid.any():
        bad_values = sorted(series[invalid].astype(str).unique().tolist())
        raise ValueError(
            f"Unexpected values in {column_name}: {bad_values}"
        )

    return converted.fillna(False).astype(bool)


def infer_recording_id(input_path: Path) -> str:
    suffix = "_rfid_anchor_assignments"
    stem = input_path.stem

    if stem.endswith(suffix):
        return stem[: -len(suffix)]

    return stem


def calculate_statistics(
    df: pd.DataFrame,
    reliable: pd.DataFrame,
    rejected: pd.DataFrame,
) -> pd.DataFrame:
    variables = [
        "nearest_distance_cm",
        "second_distance_cm",
        "nearest_second_margin_cm",
        "assigned_torso_score",
        "assigned_instance_score",
        "assigned_tracking_score",
        "assigned_n_visible",
        "frame_time_difference_ms",
    ]

    rows: list[dict[str, object]] = []

    for dataset_name, dataset in [
        ("all_provisional", df),
        ("reliable", reliable),
        ("rejected", rejected),
    ]:
        for variable in variables:
            values = pd.to_numeric(dataset[variable], errors="coerce").dropna()

            rows.append(
                {
                    "dataset": dataset_name,
                    "variable": variable,
                    "n": int(values.count()),
                    "mean": values.mean() if len(values) else np.nan,
                    "standard_deviation": values.std(ddof=1)
                    if len(values) > 1
                    else np.nan,
                    "minimum": values.min() if len(values) else np.nan,
                    "percentile_25": values.quantile(0.25)
                    if len(values)
                    else np.nan,
                    "median": values.median() if len(values) else np.nan,
                    "percentile_75": values.quantile(0.75)
                    if len(values)
                    else np.nan,
                    "maximum": values.max() if len(values) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def save_histogram(
    values: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    threshold: float | None = None,
) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(numeric, bins=25, edgecolor="black", linewidth=0.7)

    if threshold is not None:
        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"Stage 11A threshold = {threshold:g}",
        )
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RFID events")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    recording_id = args.recording_id or infer_recording_id(input_path)

    df = pd.read_csv(input_path)

    required_columns = [
        "rfid_event_index",
        "mouse_id",
        "reader",
        "assigned_track_id",
        "nearest_distance_cm",
        "second_distance_cm",
        "nearest_second_margin_cm",
        "assigned_torso_score",
        "assigned_instance_score",
        "assigned_tracking_score",
        "assigned_n_visible",
        "frame_time_difference_ms",
        "far_distance_flag",
        "ambiguity_flag",
        "far_distance_threshold_cm",
        "ambiguity_margin_threshold_cm",
    ]
    require_columns(df, required_columns)

    df["far_distance_flag"] = normalise_boolean(
        df["far_distance_flag"], "far_distance_flag"
    )
    df["ambiguity_flag"] = normalise_boolean(
        df["ambiguity_flag"], "ambiguity_flag"
    )

    df["rejection_far"] = df["far_distance_flag"]
    df["rejection_ambiguous"] = df["ambiguity_flag"]

    df["anchor_reliable"] = ~(
        df["rejection_far"] | df["rejection_ambiguous"]
    )

    conditions = [
        df["rejection_far"] & df["rejection_ambiguous"],
        df["rejection_far"] & ~df["rejection_ambiguous"],
        ~df["rejection_far"] & df["rejection_ambiguous"],
    ]
    labels = [
        "far_and_ambiguous",
        "far",
        "ambiguous",
    ]

    df["filter_status"] = np.select(
        conditions,
        labels,
        default="reliable",
    )

    reliable = df[df["anchor_reliable"]].copy()
    rejected = df[~df["anchor_reliable"]].copy()

    reliable = reliable.sort_values(
        ["nearest_frame_idx", "rfid_event_index"]
    ).reset_index(drop=True)

    rejected = rejected.sort_values(
        ["nearest_frame_idx", "rfid_event_index"]
    ).reset_index(drop=True)

    total = len(df)
    n_reliable = len(reliable)
    n_rejected = len(rejected)

    n_far_total = int(df["rejection_far"].sum())
    n_ambiguous_total = int(df["rejection_ambiguous"].sum())
    n_far_only = int(
        (df["rejection_far"] & ~df["rejection_ambiguous"]).sum()
    )
    n_ambiguous_only = int(
        (~df["rejection_far"] & df["rejection_ambiguous"]).sum()
    )
    n_both = int(
        (df["rejection_far"] & df["rejection_ambiguous"]).sum()
    )

    far_thresholds = (
        pd.to_numeric(df["far_distance_threshold_cm"], errors="coerce")
        .dropna()
        .unique()
    )
    ambiguity_thresholds = (
        pd.to_numeric(
            df["ambiguity_margin_threshold_cm"], errors="coerce"
        )
        .dropna()
        .unique()
    )

    far_threshold = (
        float(far_thresholds[0]) if len(far_thresholds) == 1 else np.nan
    )
    ambiguity_threshold = (
        float(ambiguity_thresholds[0])
        if len(ambiguity_thresholds) == 1
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "recording_id": recording_id,
                "total_provisional_assignments": total,
                "reliable_anchors": n_reliable,
                "reliable_percentage": 100 * n_reliable / total
                if total
                else np.nan,
                "rejected_total": n_rejected,
                "rejected_percentage": 100 * n_rejected / total
                if total
                else np.nan,
                "far_flag_total": n_far_total,
                "ambiguous_flag_total": n_ambiguous_total,
                "rejected_far_only": n_far_only,
                "rejected_ambiguous_only": n_ambiguous_only,
                "rejected_far_and_ambiguous": n_both,
                "far_distance_threshold_cm": far_threshold,
                "ambiguity_margin_threshold_cm": ambiguity_threshold,
                "provisional_median_distance_cm": df[
                    "nearest_distance_cm"
                ].median(),
                "reliable_median_distance_cm": reliable[
                    "nearest_distance_cm"
                ].median(),
                "provisional_median_margin_cm": df[
                    "nearest_second_margin_cm"
                ].median(),
                "reliable_median_margin_cm": reliable[
                    "nearest_second_margin_cm"
                ].median(),
                "events_inside_reader_rectangle": int(
                    df["nearest_inside_reader_rectangle"].sum()
                )
                if "nearest_inside_reader_rectangle" in df.columns
                else np.nan,
                "inside_reader_rectangle_percentage": (
                    100
                    * df["nearest_inside_reader_rectangle"].sum()
                    / total
                )
                if (
                    total
                    and "nearest_inside_reader_rectangle" in df.columns
                )
                else np.nan,
            }
        ]
    )

    statistics = calculate_statistics(df, reliable, rejected)

    reliable_path = (
        output_dir / f"{recording_id}_rfid_reliable_anchors.csv"
    )
    rejected_path = (
        output_dir / f"{recording_id}_rfid_rejected_anchors.csv"
    )
    summary_path = (
        output_dir / f"{recording_id}_rfid_anchor_filter_summary.csv"
    )
    statistics_path = (
        output_dir / f"{recording_id}_rfid_anchor_filter_statistics.csv"
    )

    reliable.to_csv(reliable_path, index=False)
    rejected.to_csv(rejected_path, index=False)
    summary.to_csv(summary_path, index=False)
    statistics.to_csv(statistics_path, index=False)

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    save_histogram(
        df["nearest_distance_cm"],
        figure_dir
        / f"{recording_id}_anchor_distance_distribution.png",
        "Distance between assigned torso and activated RFID reader",
        "Nearest-reader distance (cm)",
        threshold=far_threshold if np.isfinite(far_threshold) else None,
    )

    save_histogram(
        df["nearest_second_margin_cm"],
        figure_dir
        / f"{recording_id}_anchor_margin_distribution.png",
        "Separation between nearest and second-nearest candidates",
        "Nearest–second candidate distance margin (cm)",
        threshold=ambiguity_threshold
        if np.isfinite(ambiguity_threshold)
        else None,
    )

    save_histogram(
        df["assigned_torso_score"],
        figure_dir
        / f"{recording_id}_anchor_torso_score_distribution.png",
        "SLEAP torso confidence at provisional RFID anchors",
        "Torso confidence score",
    )

    print()
    print("Stage 11B: Reliable RFID anchor filtering")
    print("=" * 49)
    print(f"Input:                 {input_path}")
    print(f"Recording:             {recording_id}")
    print()
    print(f"Provisional anchors:   {total}")
    print(
        f"Reliable anchors:      {n_reliable} "
        f"({100 * n_reliable / total:.1f}%)"
    )
    print(
        f"Rejected anchors:      {n_rejected} "
        f"({100 * n_rejected / total:.1f}%)"
    )
    print()
    print(f"Far flag total:        {n_far_total}")
    print(f"Ambiguous flag total:  {n_ambiguous_total}")
    print(f"Far only:              {n_far_only}")
    print(f"Ambiguous only:        {n_ambiguous_only}")
    print(f"Far + ambiguous:       {n_both}")
    print()
    print(
        "Median distance:"
        f"        {df['nearest_distance_cm'].median():.3f} cm"
    )
    print(
        "Reliable median:"
        f"        {reliable['nearest_distance_cm'].median():.3f} cm"
    )
    print(
        "Median margin:"
        f"          {df['nearest_second_margin_cm'].median():.3f} cm"
    )
    print(
        "Reliable margin:"
        f"        {reliable['nearest_second_margin_cm'].median():.3f} cm"
    )
    print()
    print("Saved:")
    print(f"  Reliable anchors:    {reliable_path}")
    print(f"  Rejected anchors:    {rejected_path}")
    print(f"  Filter summary:      {summary_path}")
    print(f"  Filter statistics:   {statistics_path}")
    print(f"  Figures:             {figure_dir}")
    print()


if __name__ == "__main__":
    main()
