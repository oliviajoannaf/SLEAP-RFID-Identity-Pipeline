#!/usr/bin/env python3

"""
Analyse the RFID identity-propagation agreement-threshold sweep.

This script:
1. Loads the aggregate threshold-sweep results.
2. Loads one canonical track-summary table per recording from threshold_050.
3. Generates threshold optimisation curves.
4. Generates RFID anchor-agreement distributions.
5. Identifies tracks accepted at 50% but unresolved at 100%.
6. Exports summary tables for thesis reporting and manual validation.

The threshold_050 track summaries are used for agreement-distribution
analysis because the underlying anchor counts and agreement values do not
change between threshold runs. Using one threshold prevents the same track
from being counted eleven times.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_ROOT = Path("outputs/experimental_v2/threshold_sweep")
COMBINED_DIR = SWEEP_ROOT / "combined_results"
FIGURE_DIR = COMBINED_DIR / "figures"
TABLE_DIR = COMBINED_DIR / "validation_tables"

AGGREGATE_PATH = (
    COMBINED_DIR / "identity_threshold_sweep_aggregate.csv"
)

TRACK_SUMMARY_SUFFIX = "_identity_propagation_track_summary.csv"

THRESHOLD_LOW = 50
THRESHOLD_HIGH = 100


def check_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    source_name: str,
) -> None:
    """Raise a clear error if required columns are missing."""

    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def load_aggregate_results() -> pd.DataFrame:
    """Load aggregate threshold-sweep metrics."""

    if not AGGREGATE_PATH.exists():
        raise FileNotFoundError(
            f"Aggregate results not found: {AGGREGATE_PATH}"
        )

    aggregate = pd.read_csv(AGGREGATE_PATH)

    required = {
        "threshold_pct",
        "propagated_tracks",
        "conflicting_tracks",
        "detection_coverage_pct",
        "frame_coverage_pct",
    }

    check_required_columns(
        aggregate,
        required,
        AGGREGATE_PATH.name,
    )

    return aggregate.sort_values("threshold_pct").reset_index(drop=True)


def load_track_summaries(threshold_pct: int) -> pd.DataFrame:
    """
    Load and combine track summaries for one threshold.

    Recording names are derived from filenames.
    """

    threshold_dir = (
        SWEEP_ROOT / f"threshold_{threshold_pct:03d}"
    )

    if not threshold_dir.exists():
        raise FileNotFoundError(
            f"Threshold directory not found: {threshold_dir}"
        )

    files = sorted(
        threshold_dir.glob(f"*{TRACK_SUMMARY_SUFFIX}")
    )

    if not files:
        raise FileNotFoundError(
            f"No track-summary files found in {threshold_dir}"
        )

    frames = []

    for path in files:
        recording = path.name.removesuffix(
            TRACK_SUMMARY_SUFFIX
        )

        table = pd.read_csv(path)
        table.insert(0, "recording", recording)
        frames.append(table)

    combined = pd.concat(frames, ignore_index=True)

    required = {
        "recording",
        "track_id",
        "mouse_id",
        "identity_source",
        "identity_is_known",
        "n_rows",
        "n_unique_frames",
        "n_reliable_anchors",
        "n_dominant_anchors",
        "n_unique_mouse_ids",
        "anchor_agreement",
        "dominant_mouse_id",
    }

    check_required_columns(
        combined,
        required,
        f"threshold_{threshold_pct:03d} track summaries",
    )

    combined["identity_is_known"] = (
        combined["identity_is_known"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(combined["identity_is_known"])
        .astype(bool)
    )

    return combined


def save_figure(fig: plt.Figure, filename_stem: str) -> None:
    """Save each figure as both PNG and PDF."""

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


def plot_propagated_tracks(
    aggregate: pd.DataFrame,
) -> None:
    """Plot propagated tracks against agreement threshold."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["propagated_tracks"],
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Minimum RFID anchor agreement (%)")
    ax.set_ylabel("Tracks assigned a propagated identity")
    ax.set_title(
        "Effect of agreement threshold on identity propagation"
    )

    ax.set_xticks(
        np.arange(
            aggregate["threshold_pct"].min(),
            aggregate["threshold_pct"].max() + 1,
            5,
        )
    )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(
        fig,
        "01_propagated_tracks_vs_threshold",
    )


def plot_coverage(
    aggregate: pd.DataFrame,
) -> None:
    """Plot detection and frame coverage."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["detection_coverage_pct"],
        marker="o",
        linewidth=2,
        label="Detection coverage",
    )

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["frame_coverage_pct"],
        marker="s",
        linewidth=2,
        label="Frames with at least one known identity",
    )

    ax.set_xlabel("Minimum RFID anchor agreement (%)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(
        "Identity coverage across agreement thresholds"
    )

    ax.set_xticks(
        np.arange(
            aggregate["threshold_pct"].min(),
            aggregate["threshold_pct"].max() + 1,
            5,
        )
    )

    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_figure(
        fig,
        "02_identity_coverage_vs_threshold",
    )


def plot_unresolved_anchored_tracks(
    aggregate: pd.DataFrame,
) -> None:
    """Plot anchored tracks that remain unresolved."""

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        aggregate["threshold_pct"],
        aggregate["conflicting_tracks"],
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Minimum RFID anchor agreement (%)")
    ax.set_ylabel("Unresolved anchored tracks")
    ax.set_title(
        "Anchored tracks remaining unresolved"
    )

    ax.set_xticks(
        np.arange(
            aggregate["threshold_pct"].min(),
            aggregate["threshold_pct"].max() + 1,
            5,
        )
    )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(
        fig,
        "03_unresolved_anchored_tracks_vs_threshold",
    )


def prepare_anchored_tracks(
    low_threshold_tracks: pd.DataFrame,
) -> pd.DataFrame:
    """Retain tracks that contain at least one reliable RFID anchor."""

    anchored = low_threshold_tracks.loc[
        low_threshold_tracks["n_reliable_anchors"] > 0
    ].copy()

    anchored = anchored.loc[
        anchored["anchor_agreement"].notna()
    ].copy()

    anchored["anchor_agreement_pct"] = (
        100 * anchored["anchor_agreement"]
    )

    return anchored


def plot_agreement_histogram(
    anchored: pd.DataFrame,
) -> None:
    """Plot the distribution of dominant RFID agreement values."""

    bins = np.arange(47.5, 102.6, 5)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.hist(
        anchored["anchor_agreement_pct"],
        bins=bins,
        edgecolor="black",
    )

    ax.set_xlabel("Dominant RFID identity agreement (%)")
    ax.set_ylabel("Number of anchored tracks")
    ax.set_title(
        "Distribution of RFID anchor agreement across tracks"
    )

    ax.set_xticks(np.arange(50, 101, 5))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    save_figure(
        fig,
        "04_anchor_agreement_histogram",
    )


def plot_empirical_acceptance_curve(
    aggregate: pd.DataFrame,
) -> None:
    """
    Plot the proportion of anchored tracks accepted at each threshold.

    This uses actual threshold outputs and therefore respects tie handling
    and all implementation details.
    """

    acceptance_pct = (
        100
        * aggregate["propagated_tracks"]
        / (
            aggregate["propagated_tracks"]
            + aggregate["conflicting_tracks"]
        )
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        aggregate["threshold_pct"],
        acceptance_pct,
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Minimum RFID anchor agreement (%)")
    ax.set_ylabel("Anchored tracks accepted (%)")
    ax.set_title(
        "Empirical identity-propagation acceptance curve"
    )

    ax.set_xticks(np.arange(50, 101, 5))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(
        fig,
        "05_anchored_track_acceptance_curve",
    )


def create_agreement_summary(
    anchored: pd.DataFrame,
) -> pd.DataFrame:
    """Create summary counts for agreement ranges."""

    intervals = [
        (50, 60),
        (60, 70),
        (70, 80),
        (80, 90),
        (90, 100),
        (100, 100.000001),
    ]

    rows = []

    total = len(anchored)

    for lower, upper in intervals:
        if lower == 100:
            mask = np.isclose(
                anchored["anchor_agreement_pct"],
                100,
            )
            label = "100%"
        else:
            mask = (
                anchored["anchor_agreement_pct"].ge(lower)
                & anchored["anchor_agreement_pct"].lt(upper)
            )
            label = f"{lower}–<{upper}%"

        count = int(mask.sum())

        rows.append(
            {
                "agreement_range": label,
                "n_tracks": count,
                "percentage_of_anchored_tracks": (
                    100 * count / total if total else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def create_threshold_summary(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    """Create a concise table suitable for Results reporting."""

    output = aggregate[
        [
            "threshold_pct",
            "propagated_tracks",
            "conflicting_tracks",
            "detection_coverage_pct",
            "frame_coverage_pct",
            "additional_propagated_tracks_vs_100",
            "detection_coverage_gain_pp_vs_100",
            "frame_coverage_gain_pp_vs_100",
        ]
    ].copy()

    output = output.rename(
        columns={
            "conflicting_tracks": "unresolved_anchored_tracks",
        }
    )

    return output


def identify_newly_accepted_tracks(
    low_tracks: pd.DataFrame,
    high_tracks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify tracks known at 50% but unresolved at 100%.

    These are the tracks requiring targeted manual validation.
    """

    low_columns = [
        "recording",
        "track_id",
        "mouse_id",
        "identity_source",
        "identity_is_known",
        "n_rows",
        "n_unique_frames",
        "n_reliable_anchors",
        "n_dominant_anchors",
        "n_unique_mouse_ids",
        "anchor_agreement",
        "dominant_mouse_id",
    ]

    high_columns = [
        "recording",
        "track_id",
        "mouse_id",
        "identity_source",
        "identity_is_known",
    ]

    low = low_tracks[low_columns].copy()
    high = high_tracks[high_columns].copy()

    low = low.rename(
        columns={
            "mouse_id": "mouse_id_at_50",
            "identity_source": "identity_source_at_50",
            "identity_is_known": "known_at_50",
        }
    )

    high = high.rename(
        columns={
            "mouse_id": "mouse_id_at_100",
            "identity_source": "identity_source_at_100",
            "identity_is_known": "known_at_100",
        }
    )

    comparison = low.merge(
        high,
        on=["recording", "track_id"],
        how="inner",
        validate="one_to_one",
    )

    recovered = comparison.loc[
        comparison["known_at_50"]
        & ~comparison["known_at_100"]
    ].copy()

    recovered["anchor_agreement_pct"] = (
        100 * recovered["anchor_agreement"]
    )

    recovered = recovered.sort_values(
        [
            "anchor_agreement",
            "n_reliable_anchors",
            "n_unique_frames",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    recovered.insert(
        0,
        "validation_candidate_number",
        np.arange(1, len(recovered) + 1),
    )

    recovered["manual_assessment"] = ""
    recovered["manual_notes"] = ""
    recovered["reviewer_confidence"] = ""

    return recovered


def select_manual_validation_sample(
    recovered: pd.DataFrame,
    sample_size: int = 10,
    random_seed: int = 20260729,
) -> pd.DataFrame:
    """
    Select a reproducible sample for manual review.

    Sampling is stratified approximately across agreement values where
    possible, rather than choosing only the longest or highest-agreement
    tracks.
    """

    if recovered.empty:
        return recovered.copy()

    if len(recovered) <= sample_size:
        sample = recovered.copy()
    else:
        recovered = recovered.copy()

        recovered["agreement_band"] = pd.cut(
            recovered["anchor_agreement_pct"],
            bins=[49.999, 60, 70, 80, 90, 100.001],
            labels=[
                "50–60%",
                ">60–70%",
                ">70–80%",
                ">80–90%",
                ">90–100%",
            ],
            include_lowest=True,
        )

        samples = []

        grouped = recovered.groupby(
            "agreement_band",
            observed=True,
        )

        non_empty_groups = [
            group for _, group in grouped if not group.empty
        ]

        if non_empty_groups:
            base_per_group = max(
                1,
                sample_size // len(non_empty_groups),
            )

            for group_number, group in enumerate(
                non_empty_groups
            ):
                n_select = min(base_per_group, len(group))

                samples.append(
                    group.sample(
                        n=n_select,
                        random_state=random_seed + group_number,
                    )
                )

        sample = pd.concat(samples, ignore_index=True)

        if len(sample) < sample_size:
            already_selected = set(
                zip(sample["recording"], sample["track_id"])
            )

            remaining = recovered.loc[
                ~recovered.apply(
                    lambda row: (
                        row["recording"],
                        row["track_id"],
                    )
                    in already_selected,
                    axis=1,
                )
            ]

            n_remaining = min(
                sample_size - len(sample),
                len(remaining),
            )

            if n_remaining:
                extra = remaining.sample(
                    n=n_remaining,
                    random_state=random_seed,
                )

                sample = pd.concat(
                    [sample, extra],
                    ignore_index=True,
                )

        sample = sample.head(sample_size)

    sample = sample.sort_values(
        ["recording", "anchor_agreement", "track_id"]
    ).reset_index(drop=True)

    sample.insert(
        0,
        "manual_sample_number",
        np.arange(1, len(sample) + 1),
    )

    return sample


def print_key_results(
    aggregate: pd.DataFrame,
    anchored: pd.DataFrame,
    recovered: pd.DataFrame,
) -> None:
    """Print a concise interpretation of the generated outputs."""

    row_50 = aggregate.loc[
        aggregate["threshold_pct"] == 50
    ].iloc[0]

    row_100 = aggregate.loc[
        aggregate["threshold_pct"] == 100
    ].iloc[0]

    print()
    print("=" * 80)
    print("KEY THRESHOLD-SWEEP RESULTS")
    print("=" * 80)

    print(
        f"Propagated tracks: "
        f"{int(row_100['propagated_tracks'])} at 100% "
        f"to {int(row_50['propagated_tracks'])} at 50%."
    )

    print(
        f"Additional propagated tracks at 50%: "
        f"{int(row_50['propagated_tracks'] - row_100['propagated_tracks'])}."
    )

    print(
        f"Detection coverage: "
        f"{row_100['detection_coverage_pct']:.2f}% at 100% "
        f"to {row_50['detection_coverage_pct']:.2f}% at 50%."
    )

    print(
        f"Frame coverage: "
        f"{row_100['frame_coverage_pct']:.2f}% at 100% "
        f"to {row_50['frame_coverage_pct']:.2f}% at 50%."
    )

    print(
        f"Anchored tracks represented in agreement analysis: "
        f"{len(anchored)}."
    )

    print(
        f"Tracks newly accepted at 50% relative to 100%: "
        f"{len(recovered)}."
    )

    print()
    print(
        "The final threshold should not yet be selected. "
        "The newly accepted-track sample must first be manually reviewed."
    )


def main() -> None:
    """Run the complete analysis."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    aggregate = load_aggregate_results()

    tracks_50 = load_track_summaries(THRESHOLD_LOW)
    tracks_100 = load_track_summaries(THRESHOLD_HIGH)

    anchored = prepare_anchored_tracks(tracks_50)

    plot_propagated_tracks(aggregate)
    plot_coverage(aggregate)
    plot_unresolved_anchored_tracks(aggregate)
    plot_agreement_histogram(anchored)
    plot_empirical_acceptance_curve(aggregate)

    threshold_summary = create_threshold_summary(aggregate)
    agreement_summary = create_agreement_summary(anchored)

    recovered = identify_newly_accepted_tracks(
        tracks_50,
        tracks_100,
    )

    manual_sample = select_manual_validation_sample(
        recovered,
        sample_size=10,
        random_seed=20260729,
    )

    threshold_summary_path = (
        TABLE_DIR / "threshold_optimisation_summary.csv"
    )

    agreement_summary_path = (
        TABLE_DIR / "anchor_agreement_distribution_summary.csv"
    )

    anchored_tracks_path = (
        TABLE_DIR / "all_anchored_tracks_agreement.csv"
    )

    recovered_path = (
        TABLE_DIR / "tracks_newly_accepted_at_50_vs_100.csv"
    )

    manual_sample_path = (
        TABLE_DIR / "manual_validation_sample.csv"
    )

    threshold_summary.to_csv(
        threshold_summary_path,
        index=False,
    )

    agreement_summary.to_csv(
        agreement_summary_path,
        index=False,
    )

    anchored.to_csv(
        anchored_tracks_path,
        index=False,
    )

    recovered.to_csv(
        recovered_path,
        index=False,
    )

    manual_sample.to_csv(
        manual_sample_path,
        index=False,
    )

    print()
    print("Saved tables")
    print("=" * 80)

    for path in [
        threshold_summary_path,
        agreement_summary_path,
        anchored_tracks_path,
        recovered_path,
        manual_sample_path,
    ]:
        print(f"Saved: {path}")

    print_key_results(
        aggregate,
        anchored,
        recovered,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        raise
