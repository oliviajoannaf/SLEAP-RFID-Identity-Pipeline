#!/usr/bin/env python3

"""
Frame-level validation of RFID identity-propagation thresholds.

For every recording and agreement threshold, this script calculates:

1. Complete identity coverage:
   Percentage of analysed frames in which every expected biological
   identity is represented exactly once, all detected instances are assigned
   known identities, and no additional or duplicated identities are present.

2. Expected-identity presence:
   Percentage of analysed frames in which all expected identities are present
   at least once. This less-strict measure is retained for diagnostic use.

3. Duplicate identity rate:
   Percentage of analysed frames in which the same known mouse identity is
   assigned to more than one simultaneous SLEAP detection.

4. Unexpected identity rate:
   Percentage of frames containing a known identity that was not expected
   for that recording.

Outputs include aggregate and per-recording CSV tables and publication-quality
PNG/PDF figures.
"""

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_ROOT = Path("outputs/experimental_v2/threshold_sweep")
COMBINED_DIR = SWEEP_ROOT / "combined_results"
FIGURE_DIR = COMBINED_DIR / "figures"
TABLE_DIR = COMBINED_DIR / "validation_tables"

PROPAGATED_SUFFIX = "_identity_propagated.csv"


# Use the recording names exactly as they appear in the output filenames.
EXPECTED_MOUSE_IDS = {
    "405046407408x4miceat1244_v2": {405, 406, 407, 408},
    "405408at1020_v2": {405, 408},
    "406407408at1126_v2": {406, 407, 408},
    "4074408TE1at1040test_v2": {407, 408},
}


def parse_boolean(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations safely."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalised = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
        "nan": False,
        "none": False,
        "": False,
    }

    result = normalised.map(mapping)

    unrecognised = normalised.loc[result.isna()].unique()

    if len(unrecognised):
        raise ValueError(
            "Unrecognised identity_is_known values: "
            f"{sorted(unrecognised.tolist())}"
        )

    return result.astype(bool)


def threshold_from_directory(path: Path) -> int:
    """Extract percentage threshold from threshold_050-style directory."""

    match = re.fullmatch(r"threshold_(\d{3})", path.name)

    if not match:
        raise ValueError(
            f"Unexpected threshold directory name: {path.name}"
        )

    return int(match.group(1))


def discover_threshold_directories() -> list[Path]:
    """Find all available threshold sweep directories."""

    directories = sorted(
        path
        for path in SWEEP_ROOT.glob("threshold_*")
        if path.is_dir()
        and re.fullmatch(r"threshold_\d{3}", path.name)
    )

    if not directories:
        raise FileNotFoundError(
            f"No threshold directories found beneath {SWEEP_ROOT}"
        )

    return directories


def validate_input_columns(
    table: pd.DataFrame,
    path: Path,
) -> None:
    """Check that the propagated detection table has required columns."""

    required = {
        "frame_idx",
        "track_id",
        "mouse_id",
        "identity_is_known",
    }

    missing = required.difference(table.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )


def analyse_recording(
    path: Path,
    threshold_pct: int,
) -> tuple[dict, pd.DataFrame]:
    """Calculate frame-level identity metrics for one recording."""

    recording = path.name.removesuffix(PROPAGATED_SUFFIX)

    if recording not in EXPECTED_MOUSE_IDS:
        raise KeyError(
            f"No expected mouse-ID mapping defined for recording: "
            f"{recording}"
        )

    expected_ids = EXPECTED_MOUSE_IDS[recording]
    expected_count = len(expected_ids)

    table = pd.read_csv(path)
    validate_input_columns(table, path)

    if table.empty:
        raise ValueError(f"Input table is empty: {path}")

    table["identity_is_known"] = parse_boolean(
        table["identity_is_known"]
    )

    table["mouse_id_numeric"] = pd.to_numeric(
        table["mouse_id"],
        errors="coerce",
    )

    # A row is treated as carrying a known identity only when both the
    # boolean flag and a numeric mouse ID are available.
    table["valid_known_identity"] = (
        table["identity_is_known"]
        & table["mouse_id_numeric"].notna()
    )

    frame_rows = []

    for frame_idx, frame in table.groupby(
        "frame_idx",
        sort=True,
    ):
        known = frame.loc[
            frame["valid_known_identity"],
            "mouse_id_numeric",
        ].astype(int)

        known_counts = known.value_counts()
        known_id_set = set(known_counts.index.tolist())

        duplicate_identity = bool(
            (known_counts > 1).any()
        )

        unexpected_identity = bool(
            known_id_set.difference(expected_ids)
        )

        all_expected_present = expected_ids.issubset(
            known_id_set
        )

        every_expected_exactly_once = all(
            known_counts.get(mouse_id, 0) == 1
            for mouse_id in expected_ids
        )

        all_rows_known = bool(
            frame["valid_known_identity"].all()
        )

        # Strict definition:
        # - the number of detections equals the number of expected mice;
        # - every row has a known identity;
        # - every expected ID occurs exactly once;
        # - no duplicate or unexpected identities occur.
        complete_identity_frame = bool(
            len(frame) == expected_count
            and all_rows_known
            and every_expected_exactly_once
            and not duplicate_identity
            and not unexpected_identity
        )

        frame_rows.append(
            {
                "threshold_pct": threshold_pct,
                "recording": recording,
                "frame_idx": frame_idx,
                "expected_mouse_count": expected_count,
                "n_detections": len(frame),
                "n_known_detections": int(
                    frame["valid_known_identity"].sum()
                ),
                "n_unique_known_ids": len(known_id_set),
                "all_expected_present": all_expected_present,
                "complete_identity_frame": complete_identity_frame,
                "duplicate_identity_frame": duplicate_identity,
                "unexpected_identity_frame": unexpected_identity,
            }
        )

    frame_table = pd.DataFrame(frame_rows)

    n_frames = len(frame_table)

    summary = {
        "threshold_pct": threshold_pct,
        "recording": recording,
        "expected_mouse_ids": ";".join(
            str(mouse_id)
            for mouse_id in sorted(expected_ids)
        ),
        "expected_mouse_count": expected_count,
        "n_frames": n_frames,
        "complete_identity_frames": int(
            frame_table["complete_identity_frame"].sum()
        ),
        "complete_identity_coverage_pct": (
            100
            * frame_table["complete_identity_frame"].mean()
        ),
        "all_expected_present_frames": int(
            frame_table["all_expected_present"].sum()
        ),
        "all_expected_present_pct": (
            100
            * frame_table["all_expected_present"].mean()
        ),
        "duplicate_identity_frames": int(
            frame_table["duplicate_identity_frame"].sum()
        ),
        "duplicate_identity_rate_pct": (
            100
            * frame_table["duplicate_identity_frame"].mean()
        ),
        "unexpected_identity_frames": int(
            frame_table["unexpected_identity_frame"].sum()
        ),
        "unexpected_identity_rate_pct": (
            100
            * frame_table["unexpected_identity_frame"].mean()
        ),
    }

    return summary, frame_table


def aggregate_by_threshold(
    per_recording: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate metrics across recordings using frame-weighted percentages.

    The denominator is the total number of analysed frames across all
    recordings at each threshold.
    """

    rows = []

    for threshold_pct, group in per_recording.groupby(
        "threshold_pct",
        sort=True,
    ):
        total_frames = int(group["n_frames"].sum())

        complete_frames = int(
            group["complete_identity_frames"].sum()
        )

        all_present_frames = int(
            group["all_expected_present_frames"].sum()
        )

        duplicate_frames = int(
            group["duplicate_identity_frames"].sum()
        )

        unexpected_frames = int(
            group["unexpected_identity_frames"].sum()
        )

        rows.append(
            {
                "threshold_pct": int(threshold_pct),
                "n_recordings": int(len(group)),
                "n_frames": total_frames,
                "complete_identity_frames": complete_frames,
                "complete_identity_coverage_pct": (
                    100 * complete_frames / total_frames
                    if total_frames else np.nan
                ),
                "all_expected_present_frames": all_present_frames,
                "all_expected_present_pct": (
                    100 * all_present_frames / total_frames
                    if total_frames else np.nan
                ),
                "duplicate_identity_frames": duplicate_frames,
                "duplicate_identity_rate_pct": (
                    100 * duplicate_frames / total_frames
                    if total_frames else np.nan
                ),
                "unexpected_identity_frames": unexpected_frames,
                "unexpected_identity_rate_pct": (
                    100 * unexpected_frames / total_frames
                    if total_frames else np.nan
                ),
            }
        )

    aggregate = pd.DataFrame(rows)

    return aggregate.sort_values(
        "threshold_pct"
    ).reset_index(drop=True)


def save_figure(
    fig: plt.Figure,
    filename_stem: str,
) -> None:
    """Save a figure as PNG and vector PDF."""

    png_path = FIGURE_DIR / f"{filename_stem}.png"
    pdf_path = FIGURE_DIR / f"{filename_stem}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def configure_threshold_axis(
    ax: plt.Axes,
    aggregate: pd.DataFrame,
) -> None:
    """Apply shared threshold-axis formatting."""

    ax.set_xticks(
        aggregate["threshold_pct"].tolist()
    )

    ax.set_xlabel(
        "Minimum RFID anchor agreement (%)"
    )

    ax.grid(True, alpha=0.3)


def plot_complete_identity_coverage(
    aggregate: pd.DataFrame,
) -> None:
    """Plot strict complete identity coverage."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["complete_identity_coverage_pct"],
        marker="o",
        linewidth=2,
    )

    configure_threshold_axis(ax, aggregate)

    ax.set_ylabel(
        "Frames with complete identity assignment (%)"
    )

    ax.set_title(
        "Complete identity coverage across agreement thresholds"
    )

    ax.set_ylim(
        0,
        max(
            5,
            min(
                100,
                aggregate[
                    "complete_identity_coverage_pct"
                ].max() * 1.15,
            ),
        ),
    )

    fig.tight_layout()

    save_figure(
        fig,
        "06_complete_identity_coverage_vs_threshold",
    )


def plot_duplicate_identity_rate(
    aggregate: pd.DataFrame,
) -> None:
    """Plot the proportion of frames containing duplicated identities."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["duplicate_identity_rate_pct"],
        marker="o",
        linewidth=2,
    )

    configure_threshold_axis(ax, aggregate)

    ax.set_ylabel(
        "Frames containing a duplicated identity (%)"
    )

    ax.set_title(
        "Duplicate identity rate across agreement thresholds"
    )

    lower = aggregate[
        "duplicate_identity_rate_pct"
    ].min()

    upper = aggregate[
        "duplicate_identity_rate_pct"
    ].max()

    padding = max(
        0.1,
        (upper - lower) * 0.20,
    )

    ax.set_ylim(
        max(0, lower - padding),
        min(100, upper + padding),
    )

    fig.tight_layout()

    save_figure(
        fig,
        "07_duplicate_identity_rate_vs_threshold",
    )


def plot_combined_validation_metrics(
    aggregate: pd.DataFrame,
) -> None:
    """Plot coverage and duplicate rates on one common percentage axis."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["complete_identity_coverage_pct"],
        marker="o",
        linewidth=2,
        label="Complete identity coverage",
    )

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["duplicate_identity_rate_pct"],
        marker="s",
        linewidth=2,
        label="Duplicate identity rate",
    )

    configure_threshold_axis(ax, aggregate)

    ax.set_ylabel("Frames (%)")

    ax.set_title(
        "Frame-level identity validation across agreement thresholds"
    )

    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        "08_frame_level_identity_validation_vs_threshold",
    )


def print_key_results(
    aggregate: pd.DataFrame,
) -> None:
    """Print comparison between the lowest and highest thresholds."""

    low = aggregate.iloc[0]
    high = aggregate.iloc[-1]

    print()
    print("=" * 80)
    print("FRAME-LEVEL THRESHOLD VALIDATION")
    print("=" * 80)

    print(
        f"Thresholds analysed: "
        f"{int(low['threshold_pct'])}% to "
        f"{int(high['threshold_pct'])}%."
    )

    print(
        f"Frames analysed per threshold: "
        f"{int(low['n_frames'])}."
    )

    print()
    print(
        "Complete identity coverage:"
    )

    print(
        f"  {int(high['threshold_pct'])}% threshold: "
        f"{high['complete_identity_coverage_pct']:.4f}%"
    )

    print(
        f"  {int(low['threshold_pct'])}% threshold: "
        f"{low['complete_identity_coverage_pct']:.4f}%"
    )

    print(
        f"  Change: "
        f"{low['complete_identity_coverage_pct'] - high['complete_identity_coverage_pct']:+.4f} "
        "percentage points"
    )

    print()
    print(
        "Duplicate identity rate:"
    )

    print(
        f"  {int(high['threshold_pct'])}% threshold: "
        f"{high['duplicate_identity_rate_pct']:.4f}%"
    )

    print(
        f"  {int(low['threshold_pct'])}% threshold: "
        f"{low['duplicate_identity_rate_pct']:.4f}%"
    )

    print(
        f"  Change: "
        f"{low['duplicate_identity_rate_pct'] - high['duplicate_identity_rate_pct']:+.4f} "
        "percentage points"
    )

    print()
    print(
        "These automated metrics should be interpreted alongside the "
        "targeted manual review before selecting the production threshold."
    )


def main() -> None:
    """Run frame-level validation across all thresholds."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    frame_tables = []

    threshold_directories = discover_threshold_directories()

    for threshold_dir in threshold_directories:
        threshold_pct = threshold_from_directory(
            threshold_dir
        )

        files = sorted(
            threshold_dir.glob(
                f"*{PROPAGATED_SUFFIX}"
            )
        )

        if not files:
            raise FileNotFoundError(
                f"No propagated detection CSVs found in "
                f"{threshold_dir}"
            )

        found_recordings = {
            path.name.removesuffix(PROPAGATED_SUFFIX)
            for path in files
        }

        expected_recordings = set(
            EXPECTED_MOUSE_IDS
        )

        missing_recordings = (
            expected_recordings - found_recordings
        )

        unexpected_recordings = (
            found_recordings - expected_recordings
        )

        if missing_recordings:
            raise FileNotFoundError(
                f"Threshold {threshold_pct}% is missing recordings: "
                f"{sorted(missing_recordings)}"
            )

        if unexpected_recordings:
            raise KeyError(
                f"Threshold {threshold_pct}% contains unmapped recordings: "
                f"{sorted(unexpected_recordings)}"
            )

        for path in files:
            summary, frame_table = analyse_recording(
                path,
                threshold_pct,
            )

            summaries.append(summary)
            frame_tables.append(frame_table)

        print(
            f"Analysed threshold {threshold_pct}%: "
            f"{len(files)} recordings"
        )

    per_recording = pd.DataFrame(summaries)

    per_recording = per_recording.sort_values(
        ["threshold_pct", "recording"]
    ).reset_index(drop=True)

    aggregate = aggregate_by_threshold(
        per_recording
    )

    all_frames = pd.concat(
        frame_tables,
        ignore_index=True,
    )

    per_recording_path = (
        TABLE_DIR
        / "frame_identity_validation_per_recording.csv"
    )

    aggregate_path = (
        TABLE_DIR
        / "frame_identity_validation_aggregate.csv"
    )

    all_frames_path = (
        TABLE_DIR
        / "frame_identity_validation_all_frames.csv"
    )

    per_recording.to_csv(
        per_recording_path,
        index=False,
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    all_frames.to_csv(
        all_frames_path,
        index=False,
    )

    print()
    print(f"Saved: {per_recording_path}")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {all_frames_path}")

    plot_complete_identity_coverage(aggregate)
    plot_duplicate_identity_rate(aggregate)
    plot_combined_validation_metrics(aggregate)

    print_key_results(aggregate)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        raise
