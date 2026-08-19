#!/usr/bin/env python3

"""
Stage 12: Propagate RFID identities across SLEAP track fragments.

Rules
-----
1. The dominant RFID identity and its agreement proportion are calculated
   independently for each SLEAP track.
2. A biological identity is propagated only when the dominant identity is
   unique and its agreement meets the specified minimum threshold.
3. Tied or below-threshold tracks remain unknown.
4. Tracks without reliable RFID anchors remain unknown.
5. The original SLEAP table is preserved, with identity columns appended.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RECORDING = "407405spareat1313"
DEFAULT_OUTPUT_ROOT = Path("outputs/development_v1")
DEFAULT_IDENTITY_DIR = DEFAULT_OUTPUT_ROOT / "identity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate reliable RFID identities across SLEAP tracks."
    )

    parser.add_argument(
        "--recording",
        default=DEFAULT_RECORDING,
        help="Recording identifier.",
    )

    parser.add_argument(
        "--anchors",
        type=Path,
        default=None,
        help="Reliable RFID anchor CSV.",
    )

    parser.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="Complete SLEAP tracking/detection CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_IDENTITY_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--min-agreement",
        type=float,
        default=1.0,
        help=(
            "Minimum dominant-anchor agreement required for identity "
            "propagation, expressed as a proportion from 0.5 to 1.0. "
            "Default: 1.0."
        ),
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


def locate_detection_file(
    root: Path,
    recording: str,
) -> Path:
    """
    Search for the complete SLEAP table.

    Candidate files must:
    - contain the recording name;
    - contain an assigned track column;
    - contain a frame column;
    - not be an RFID anchor or summary output.

    The largest suitable CSV is selected because the full detection table
    should normally contain substantially more rows than summary tables.
    """

    excluded_terms = (
        "rfid_reliable_anchors",
        "anchor_track_summary",
        "anchor_mouse_summary",
        "anchor_reader_summary",
        "anchor_time_summary",
        "anchor_validation_summary",
        "identity_summary",
        "identity_propagated",
    )

    candidates: list[tuple[int, Path]] = []

    for path in root.rglob(f"*{recording}*.csv"):
        lower_name = path.name.lower()

        if any(term in lower_name for term in excluded_terms):
            continue

        try:
            preview = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        has_track = any(
            column in preview.columns
            for column in (
                "assigned_track_id",
                "track_id",
                "track",
                "track_name",
            )
        )

        has_frame = any(
            column in preview.columns
            for column in (
                "frame_idx",
                "frame",
                "frame_index",
                "video_frame",
            )
        )

        if not (has_track and has_frame):
            continue

        try:
            row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        except Exception:
            row_count = 0

        candidates.append((row_count, path))

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically locate the complete SLEAP CSV.\n"
            "Run the script again with:\n"
            "  --detections PATH_TO_COMPLETE_SLEAP_CSV"
        )

    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def load_inputs(
    anchors_path: Path,
    detections_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not anchors_path.exists():
        raise FileNotFoundError(
            f"Reliable-anchor file not found: {anchors_path}"
        )

    if not detections_path.exists():
        raise FileNotFoundError(
            f"SLEAP detection file not found: {detections_path}"
        )

    anchors = pd.read_csv(anchors_path)
    detections = pd.read_csv(detections_path)

    if anchors.empty:
        raise ValueError("Reliable-anchor table is empty.")

    if detections.empty:
        raise ValueError("SLEAP detection table is empty.")

    return anchors, detections


def normalise_track_column(
    df: pd.DataFrame,
    track_column: str,
) -> pd.DataFrame:
    result = df.copy()

    result[track_column] = result[track_column].astype("string").str.strip()

    result.loc[
        result[track_column].isin(["", "nan", "None", "<NA>"]),
        track_column,
    ] = pd.NA

    return result


def build_track_identity_lookup(
    anchors: pd.DataFrame,
    anchor_track_column: str,
    mouse_column: str,
    min_agreement: float,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Calculate the dominant RFID identity and agreement for every anchored track.

    A track is eligible for propagation when:
    - one identity has a strictly greater anchor count than every other identity;
    - dominant-anchor agreement is at least ``min_agreement``.

    Exact ties remain unresolved, including at a threshold of 0.50.
    """

    working = anchors[
        [
            anchor_track_column,
            mouse_column,
            "nearest_frame_idx",
        ]
    ].dropna(
        subset=[
            anchor_track_column,
            mouse_column,
        ]
    ).copy()

    working[anchor_track_column] = (
        working[anchor_track_column]
        .astype("string")
        .str.strip()
    )

    rows: list[dict[str, object]] = []

    for track_id, track_anchors in working.groupby(
        anchor_track_column,
        dropna=False,
    ):
        track_anchors = track_anchors.sort_values(
            "nearest_frame_idx",
            kind="stable",
        )

        anchor_sequence = ",".join(
            track_anchors[mouse_column].astype(str)
        )

        anchor_frame_sequence = ",".join(
            pd.to_numeric(
                track_anchors["nearest_frame_idx"],
                errors="coerce",
            )
            .astype("Int64")
            .astype(str)
        )

        identity_counts = track_anchors[mouse_column].value_counts()

        dominant_mouse_id = identity_counts.index[0]
        dominant_count = int(identity_counts.iloc[0])
        total_count = int(identity_counts.sum())
        n_unique_mouse_ids = int(identity_counts.size)
        agreement = dominant_count / total_count

        if len(identity_counts) == 1:
            unique_dominant_identity = True
        else:
            unique_dominant_identity = (
                dominant_count > int(identity_counts.iloc[1])
            )

        meets_threshold = (
            unique_dominant_identity
            and agreement >= min_agreement
        )

        rows.append(
            {
                anchor_track_column: track_id,
                "propagated_mouse_id": (
                    dominant_mouse_id if meets_threshold else pd.NA
                ),
                "dominant_mouse_id": dominant_mouse_id,
                "n_reliable_anchors": total_count,
                "n_dominant_anchors": dominant_count,
                "n_unique_mouse_ids": n_unique_mouse_ids,
                "anchor_sequence": anchor_sequence,
                "anchor_frame_sequence": anchor_frame_sequence,
                "anchor_agreement": agreement,
                "meets_agreement_threshold": meets_threshold,
                "has_unique_dominant_identity": unique_dominant_identity,
            }
        )

    lookup = pd.DataFrame(rows)

    conflicting_tracks = set(
        lookup.loc[
            ~lookup["meets_agreement_threshold"],
            anchor_track_column,
        ]
        .dropna()
        .astype(str)
    )

    return lookup, conflicting_tracks

def propagate_identities(
    detections: pd.DataFrame,
    lookup: pd.DataFrame,
    conflicting_tracks: set[str],
    detection_track_column: str,
    anchor_track_column: str,
) -> pd.DataFrame:
    result = detections.copy()

    if detection_track_column != anchor_track_column:
        lookup = lookup.rename(
            columns={anchor_track_column: detection_track_column}
        )

    result = result.merge(
        lookup,
        on=detection_track_column,
        how="left",
        validate="many_to_one",
    )

    track_strings = result[detection_track_column].astype("string")

    result["mouse_id"] = pd.to_numeric(
        result["propagated_mouse_id"],
        errors="coerce",
    ).astype("Int64")

    result["identity_source"] = "no_rfid_anchor"

    missing_track = result[detection_track_column].isna()
    conflicting = track_strings.isin(conflicting_tracks)
    propagated = result["mouse_id"].notna()

    result.loc[missing_track, "identity_source"] = "no_track"
    result.loc[conflicting, "identity_source"] = "conflicting_anchors"
    result.loc[propagated, "identity_source"] = "rfid_propagated"

    result["identity_is_known"] = propagated

    result = result.drop(columns=["propagated_mouse_id"])

    return result


def create_track_summary(
    propagated: pd.DataFrame,
    track_column: str,
    frame_column: str,
) -> pd.DataFrame:
    valid_tracks = propagated[
        propagated[track_column].notna()
    ].copy()

    summary = (
        valid_tracks.groupby(track_column, dropna=False)
        .agg(
            mouse_id=("mouse_id", "first"),
            identity_source=("identity_source", "first"),
            identity_is_known=("identity_is_known", "max"),
            n_rows=(track_column, "size"),
            n_unique_frames=(frame_column, "nunique"),
            n_reliable_anchors=(
                "n_reliable_anchors",
                lambda values: int(values.dropna().max())
                if values.notna().any()
                else 0,
            ),
            n_dominant_anchors=(
                "n_dominant_anchors",
                lambda values: int(values.dropna().max())
                if values.notna().any()
                else 0,
            ),
            n_unique_mouse_ids=(
                "n_unique_mouse_ids",
                lambda values: int(values.dropna().max())
                if values.notna().any()
                else 0,
            ),
            anchor_agreement=(
                "anchor_agreement",
                lambda values: float(values.dropna().max())
                if values.notna().any()
                else pd.NA,
            ),
            dominant_mouse_id=(
                "dominant_mouse_id",
                lambda values: values.dropna().iloc[0]
                if values.notna().any()
                else pd.NA,
            ),
            anchor_sequence=(
                "anchor_sequence",
                lambda values: values.dropna().iloc[0]
                if values.notna().any()
                else pd.NA,
            ),
            anchor_frame_sequence=(
                "anchor_frame_sequence",
                lambda values: values.dropna().iloc[0]
                if values.notna().any()
                else pd.NA,
            ),
        )
        .reset_index()
    )

    return summary.sort_values(
        ["identity_is_known", track_column],
        ascending=[False, True],
    )


def create_validation_summary(
    propagated: pd.DataFrame,
    track_summary: pd.DataFrame,
    frame_column: str,
) -> pd.DataFrame:
    total_rows = len(propagated)
    known_rows = int(propagated["identity_is_known"].sum())
    unknown_rows = total_rows - known_rows

    total_frames = propagated[frame_column].nunique()

    known_frames = propagated.loc[
        propagated["identity_is_known"],
        frame_column,
    ].nunique()

    metrics = [
        ("total_detection_rows", total_rows),
        ("identity_labelled_rows", known_rows),
        ("unknown_rows", unknown_rows),
        (
            "percentage_rows_identity_labelled",
            100.0 * known_rows / total_rows if total_rows else 0.0,
        ),
        ("total_unique_video_frames", total_frames),
        ("frames_with_at_least_one_known_identity", known_frames),
        (
            "percentage_frames_with_known_identity",
            100.0 * known_frames / total_frames if total_frames else 0.0,
        ),
        ("total_tracks", len(track_summary)),
        (
            "tracks_with_propagated_identity",
            int(track_summary["identity_is_known"].sum()),
        ),
        (
            "tracks_with_conflicting_anchors",
            int(
                (
                    track_summary["identity_source"]
                    == "conflicting_anchors"
                ).sum()
            ),
        ),
        (
            "tracks_without_rfid_anchor",
            int(
                (
                    track_summary["identity_source"]
                    == "no_rfid_anchor"
                ).sum()
            ),
        ),
    ]

    return pd.DataFrame(metrics, columns=["metric", "value"])


def main() -> None:
    args = parse_args()

    if not 0.5 <= args.min_agreement <= 1.0:
        raise ValueError(
            "--min-agreement must be between 0.5 and 1.0."
        )

    anchors_path = args.anchors or (
        args.output_dir
        / f"{args.recording}_rfid_reliable_anchors.csv"
    )

    if args.detections is None:
        detections_path = locate_detection_file(
            DEFAULT_OUTPUT_ROOT,
            args.recording,
        )
    else:
        detections_path = args.detections

    print()
    print("Stage 12: RFID identity propagation")
    print("=" * 43)
    print(f"Reliable anchors: {anchors_path}")
    print(f"SLEAP detections: {detections_path}")
    print(
        "Minimum agreement: "
        f"{100.0 * args.min_agreement:.1f}%"
    )
    print()

    anchors, detections = load_inputs(
        anchors_path,
        detections_path,
    )

    anchor_track_column = find_column(
        anchors,
        [
            "assigned_track_id",
            "track_id",
            "track",
            "track_name",
        ],
        "track column in reliable anchors",
    )

    detection_track_column = find_column(
        detections,
        [
            "assigned_track_id",
            "track_id",
            "track",
            "track_name",
        ],
        "track column in the SLEAP table",
    )

    mouse_column = find_column(
        anchors,
        ["mouse_id", "rfid_mouse_id", "animal_id"],
        "mouse identity column",
    )

    frame_column = find_column(
        detections,
        [
            "frame_idx",
            "frame",
            "frame_index",
            "video_frame",
        ],
        "frame column",
    )

    anchors = normalise_track_column(
        anchors,
        anchor_track_column,
    )

    detections = normalise_track_column(
        detections,
        detection_track_column,
    )

    lookup, conflicting_tracks = build_track_identity_lookup(
        anchors,
        anchor_track_column,
        mouse_column,
        args.min_agreement,
    )

    propagated = propagate_identities(
        detections,
        lookup,
        conflicting_tracks,
        detection_track_column,
        anchor_track_column,
    )

    track_summary = create_track_summary(
        propagated,
        detection_track_column,
        frame_column,
    )

    validation_summary = create_validation_summary(
        propagated,
        track_summary,
        frame_column,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    propagated_path = (
        args.output_dir
        / f"{args.recording}_identity_propagated.csv"
    )

    track_summary_path = (
        args.output_dir
        / f"{args.recording}_identity_propagation_track_summary.csv"
    )

    validation_summary_path = (
        args.output_dir
        / f"{args.recording}_identity_propagation_summary.csv"
    )

    propagated.to_csv(propagated_path, index=False)
    track_summary.to_csv(track_summary_path, index=False)
    validation_summary.to_csv(validation_summary_path, index=False)

    summary_values = dict(
        zip(
            validation_summary["metric"],
            validation_summary["value"],
        )
    )

    print(f"Total tracks:                 {int(summary_values['total_tracks'])}")
    print(
        "Tracks assigned identity:     "
        f"{int(summary_values['tracks_with_propagated_identity'])}"
    )
    print(
        "Conflicting anchored tracks:  "
        f"{int(summary_values['tracks_with_conflicting_anchors'])}"
    )
    print(
        "Tracks without RFID anchors:  "
        f"{int(summary_values['tracks_without_rfid_anchor'])}"
    )
    print()
    print(
        "Detection rows labelled:      "
        f"{int(summary_values['identity_labelled_rows'])}/"
        f"{int(summary_values['total_detection_rows'])} "
        f"({summary_values['percentage_rows_identity_labelled']:.2f}%)"
    )
    print(
        "Frames with known identity:   "
        f"{int(summary_values['frames_with_at_least_one_known_identity'])}/"
        f"{int(summary_values['total_unique_video_frames'])} "
        f"({summary_values['percentage_frames_with_known_identity']:.2f}%)"
    )
    print()
    print("Saved:")
    print(f"  Identity-labelled table: {propagated_path}")
    print(f"  Track summary:           {track_summary_path}")
    print(f"  Validation summary:      {validation_summary_path}")


if __name__ == "__main__":
    main()
