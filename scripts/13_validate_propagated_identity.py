#!/usr/bin/env python3

"""
Stage 13: Validate propagated RFID identities.

Checks
------
1. Whether the same mouse is assigned to multiple tracks in one frame.
2. Whether same-mouse track fragments overlap temporally.
3. Identity coverage by track, detection row and video frame.
4. First/last frame and duration of each track fragment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RECORDING = "407405spareat1313"
DEFAULT_IDENTITY_DIR = Path("outputs/development_v1/identity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate propagated RFID identities."
    )

    parser.add_argument(
        "--recording",
        default=DEFAULT_RECORDING,
        help="Recording identifier.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Identity-propagated CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_IDENTITY_DIR,
        help="Directory for validation outputs.",
    )

    return parser.parse_args()


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description}. "
        f"Expected one of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_identity_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Identity-propagated table not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Identity-propagated table is empty.")

    return df


def prepare_table(
    df: pd.DataFrame,
    frame_column: str,
    track_column: str,
) -> pd.DataFrame:
    result = df.copy()

    result[frame_column] = pd.to_numeric(
        result[frame_column],
        errors="coerce",
    ).astype("Int64")

    result[track_column] = (
        result[track_column]
        .astype("string")
        .str.strip()
    )

    result.loc[
        result[track_column].isin(["", "nan", "None", "<NA>"]),
        track_column,
    ] = pd.NA

    result["mouse_id"] = pd.to_numeric(
        result["mouse_id"],
        errors="coerce",
    ).astype("Int64")

    if "identity_is_known" not in result.columns:
        result["identity_is_known"] = result["mouse_id"].notna()
    else:
        result["identity_is_known"] = (
            result["identity_is_known"]
            .astype("string")
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
            .fillna(result["mouse_id"].notna())
            .astype(bool)
        )

    return result


def find_duplicate_identity_frames(
    df: pd.DataFrame,
    frame_column: str,
    track_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find frames where the same mouse is assigned to multiple tracks.
    """

    known = df[
        df["identity_is_known"]
        & df["mouse_id"].notna()
        & df[track_column].notna()
        & df[frame_column].notna()
    ].copy()

    grouped = (
        known.groupby(
            [frame_column, "mouse_id"],
            dropna=False,
        )
        .agg(
            n_tracks=(track_column, "nunique"),
            tracks=(
                track_column,
                lambda values: "|".join(
                    sorted(set(values.astype(str)))
                ),
            ),
            n_detection_rows=(track_column, "size"),
        )
        .reset_index()
    )

    duplicate_groups = grouped[
        grouped["n_tracks"] > 1
    ].copy()

    if duplicate_groups.empty:
        duplicate_rows = known.iloc[0:0].copy()
    else:
        duplicate_rows = known.merge(
            duplicate_groups[
                [frame_column, "mouse_id"]
            ],
            on=[frame_column, "mouse_id"],
            how="inner",
        )

        duplicate_rows = duplicate_rows.sort_values(
            [frame_column, "mouse_id", track_column]
        )

    return duplicate_groups, duplicate_rows


def create_track_continuity_summary(
    df: pd.DataFrame,
    frame_column: str,
    track_column: str,
) -> pd.DataFrame:
    valid_tracks = df[
        df[track_column].notna()
        & df[frame_column].notna()
    ].copy()

    summary = (
        valid_tracks.groupby(
            track_column,
            dropna=False,
        )
        .agg(
            mouse_id=("mouse_id", "first"),
            identity_source=("identity_source", "first"),
            identity_is_known=("identity_is_known", "max"),
            first_frame=(frame_column, "min"),
            last_frame=(frame_column, "max"),
            n_unique_frames=(frame_column, "nunique"),
            n_detection_rows=(track_column, "size"),
        )
        .reset_index()
    )

    summary["frame_span"] = (
        summary["last_frame"]
        - summary["first_frame"]
        + 1
    )

    summary["frame_coverage_within_span_percent"] = (
        100.0
        * summary["n_unique_frames"]
        / summary["frame_span"]
    )

    return summary.sort_values(
        ["mouse_id", "first_frame", track_column],
        na_position="last",
    )


def find_same_mouse_track_overlaps(
    track_summary: pd.DataFrame,
    track_column: str,
) -> pd.DataFrame:
    known = track_summary[
        track_summary["identity_is_known"]
        & track_summary["mouse_id"].notna()
    ].copy()

    records: list[dict[str, object]] = []

    for mouse_id, group in known.groupby("mouse_id"):
        rows = group.sort_values("first_frame").to_dict("records")

        for index, first in enumerate(rows):
            for second in rows[index + 1:]:
                overlap_start = max(
                    int(first["first_frame"]),
                    int(second["first_frame"]),
                )

                overlap_end = min(
                    int(first["last_frame"]),
                    int(second["last_frame"]),
                )

                if overlap_start <= overlap_end:
                    records.append(
                        {
                            "mouse_id": int(mouse_id),
                            "track_a": first[track_column],
                            "track_b": second[track_column],
                            "track_a_first_frame": int(
                                first["first_frame"]
                            ),
                            "track_a_last_frame": int(
                                first["last_frame"]
                            ),
                            "track_b_first_frame": int(
                                second["first_frame"]
                            ),
                            "track_b_last_frame": int(
                                second["last_frame"]
                            ),
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_frames": (
                                overlap_end
                                - overlap_start
                                + 1
                            ),
                        }
                    )

    columns = [
        "mouse_id",
        "track_a",
        "track_b",
        "track_a_first_frame",
        "track_a_last_frame",
        "track_b_first_frame",
        "track_b_last_frame",
        "overlap_start",
        "overlap_end",
        "overlap_frames",
    ]

    return pd.DataFrame(records, columns=columns)


def create_validation_summary(
    df: pd.DataFrame,
    track_summary: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
    overlaps: pd.DataFrame,
    frame_column: str,
) -> pd.DataFrame:
    total_rows = len(df)
    known_rows = int(df["identity_is_known"].sum())

    total_frames = int(df[frame_column].nunique())

    known_frames = int(
        df.loc[
            df["identity_is_known"],
            frame_column,
        ].nunique()
    )

    duplicate_frame_count = int(
        duplicate_groups[frame_column].nunique()
        if not duplicate_groups.empty
        else 0
    )

    duplicate_mouse_frame_groups = len(duplicate_groups)

    duplicate_frame_percent = (
        100.0 * duplicate_frame_count / total_frames
        if total_frames
        else 0.0
    )

    metrics = [
        ("total_detection_rows", total_rows),
        ("identity_labelled_rows", known_rows),
        (
            "percentage_detection_rows_labelled",
            100.0 * known_rows / total_rows
            if total_rows
            else 0.0,
        ),
        ("total_video_frames", total_frames),
        ("frames_with_known_identity", known_frames),
        (
            "percentage_frames_with_known_identity",
            100.0 * known_frames / total_frames
            if total_frames
            else 0.0,
        ),
        ("total_tracks", len(track_summary)),
        (
            "identity_labelled_tracks",
            int(track_summary["identity_is_known"].sum()),
        ),
        (
            "unknown_tracks",
            int((~track_summary["identity_is_known"]).sum()),
        ),
        (
            "duplicate_identity_frames",
            duplicate_frame_count,
        ),
        (
            "duplicate_mouse_frame_groups",
            duplicate_mouse_frame_groups,
        ),
        (
            "percentage_frames_with_duplicate_identity",
            duplicate_frame_percent,
        ),
        (
            "same_mouse_overlapping_track_pairs",
            len(overlaps),
        ),
        (
            "maximum_same_mouse_overlap_frames",
            int(overlaps["overlap_frames"].max())
            if not overlaps.empty
            else 0,
        ),
    ]

    return pd.DataFrame(
        metrics,
        columns=["metric", "value"],
    )


def main() -> None:
    args = parse_args()

    input_path = args.input or (
        args.output_dir
        / f"{args.recording}_identity_propagated.csv"
    )

    print()
    print("Stage 13: Propagated identity validation")
    print("=" * 42)
    print(f"Input: {input_path}")
    print()

    df = load_identity_table(input_path)

    frame_column = find_column(
        df,
        [
            "frame_idx",
            "frame",
            "frame_index",
            "video_frame",
        ],
        "frame column",
    )

    track_column = find_column(
        df,
        [
            "assigned_track_id",
            "track_id",
            "track",
            "track_name",
        ],
        "track column",
    )

    find_column(
        df,
        ["mouse_id"],
        "propagated mouse identity column",
    )

    if "identity_source" not in df.columns:
        raise ValueError(
            "Missing identity_source column. "
            "Run Stage 12 before Stage 13."
        )

    df = prepare_table(
        df,
        frame_column,
        track_column,
    )

    duplicate_groups, duplicate_rows = (
        find_duplicate_identity_frames(
            df,
            frame_column,
            track_column,
        )
    )

    track_summary = create_track_continuity_summary(
        df,
        frame_column,
        track_column,
    )

    overlaps = find_same_mouse_track_overlaps(
        track_summary,
        track_column,
    )

    validation_summary = create_validation_summary(
        df,
        track_summary,
        duplicate_groups,
        overlaps,
        frame_column,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    duplicate_groups_path = (
        args.output_dir
        / f"{args.recording}_identity_duplicate_frame_summary.csv"
    )

    duplicate_rows_path = (
        args.output_dir
        / f"{args.recording}_identity_duplicate_frame_rows.csv"
    )

    track_summary_path = (
        args.output_dir
        / f"{args.recording}_identity_track_continuity_summary.csv"
    )

    overlaps_path = (
        args.output_dir
        / f"{args.recording}_identity_track_overlap_summary.csv"
    )

    validation_path = (
        args.output_dir
        / f"{args.recording}_identity_final_validation_summary.csv"
    )

    duplicate_groups.to_csv(
        duplicate_groups_path,
        index=False,
    )

    duplicate_rows.to_csv(
        duplicate_rows_path,
        index=False,
    )

    track_summary.to_csv(
        track_summary_path,
        index=False,
    )

    overlaps.to_csv(
        overlaps_path,
        index=False,
    )

    validation_summary.to_csv(
        validation_path,
        index=False,
    )

    values = dict(
        zip(
            validation_summary["metric"],
            validation_summary["value"],
        )
    )

    print(
        "Identity-labelled rows:       "
        f"{int(values['identity_labelled_rows'])}/"
        f"{int(values['total_detection_rows'])} "
        f"({values['percentage_detection_rows_labelled']:.2f}%)"
    )

    print(
        "Frames with known identity:   "
        f"{int(values['frames_with_known_identity'])}/"
        f"{int(values['total_video_frames'])} "
        f"({values['percentage_frames_with_known_identity']:.2f}%)"
    )

    print()
    print(
        "Duplicate identity frames:    "
        f"{int(values['duplicate_identity_frames'])} "
        f"({values['percentage_frames_with_duplicate_identity']:.4f}%)"
    )

    print(
        "Duplicate mouse-frame groups: "
        f"{int(values['duplicate_mouse_frame_groups'])}"
    )

    print(
        "Same-mouse overlap pairs:     "
        f"{int(values['same_mouse_overlapping_track_pairs'])}"
    )

    print(
        "Maximum overlap duration:     "
        f"{int(values['maximum_same_mouse_overlap_frames'])} frames"
    )

    print()

    if int(values["duplicate_identity_frames"]) == 0:
        print("PASS: No mouse was assigned to multiple tracks in one frame.")
    else:
        print(
            "WARNING: Duplicate identity assignments were detected. "
            "Inspect the duplicate-frame outputs before downstream analysis."
        )

    print()
    print("Saved:")
    print(f"  Final summary:       {validation_path}")
    print(f"  Track continuity:    {track_summary_path}")
    print(f"  Track overlaps:      {overlaps_path}")
    print(f"  Duplicate summary:   {duplicate_groups_path}")
    print(f"  Duplicate rows:      {duplicate_rows_path}")


if __name__ == "__main__":
    main()
