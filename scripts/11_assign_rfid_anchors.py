#!/usr/bin/env python3

"""
Stage 11A: Assign provisional RFID identity anchors to SLEAP tracks.

For each temporally aligned RFID event:
1. Find all SLEAP instances in the nearest video frame.
2. Use torso_centre_x and torso_centre_y as the RFID-tag position proxy.
3. Calculate each track's distance from the active RFID reader.
4. Rank candidate tracks.
5. Save the nearest track as a provisional RFID anchor.
6. Retain spatial evidence and diagnostic flags.

This stage does NOT propagate identities through time.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VIDEO_ID = "407405spareat1313"

DEFAULT_SLEAP = Path(
    "outputs/development_v1/tables/"
    "407405spareat1313_timestamped.csv"
)

DEFAULT_RFID = Path(
    "outputs/development_v1/rfid/"
    "407405spareat1313_rfid_video_aligned.csv"
)

DEFAULT_READER_CM = Path(
    "config/rfid_reader_positions_cm.csv"
)

DEFAULT_CALIBRATION = Path(
    "config/arena_calibration_407405spareat1313.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/development_v1/identity"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign provisional RFID events to the nearest SLEAP "
            "track using torso-centre coordinates."
        )
    )

    parser.add_argument(
        "--video-id",
        default=DEFAULT_VIDEO_ID,
        help="Video identifier used in output filenames.",
    )

    parser.add_argument(
        "--sleap",
        type=Path,
        default=DEFAULT_SLEAP,
        help="Timestamped SLEAP CSV.",
    )

    parser.add_argument(
        "--rfid",
        type=Path,
        default=DEFAULT_RFID,
        help="Stage 09 aligned RFID CSV.",
    )

    parser.add_argument(
        "--reader-cm",
        type=Path,
        default=DEFAULT_READER_CM,
        help="RFID reader positions and dimensions in centimetres.",
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Arena pixel calibration CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 11A outputs.",
    )

    parser.add_argument(
        "--ambiguity-margin-cm",
        type=float,
        default=2.0,
        help=(
            "Flag an event as spatially ambiguous when the difference "
            "between the nearest and second-nearest tracks is below "
            "this value. This is a diagnostic threshold, not a final "
            "acceptance rule."
        ),
    )

    parser.add_argument(
        "--far-distance-cm",
        type=float,
        default=8.0,
        help=(
            "Flag the nearest candidate as spatially distant when its "
            "torso centre is farther than this value from the reader "
            "centre. This is a diagnostic threshold."
        ),
    )

    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def load_calibration(path: Path) -> dict[str, float]:
    calibration = pd.read_csv(path)

    require_columns(
        calibration,
        [
            "arena_width_cm",
            "arena_height_cm",
            "left_px",
            "right_px",
            "top_px",
            "bottom_px",
        ],
        "Arena calibration",
    )

    if len(calibration) != 1:
        raise ValueError(
            "Arena calibration must contain exactly one row. "
            f"Found {len(calibration)} rows."
        )

    row = calibration.iloc[0]

    arena_width_cm = float(row["arena_width_cm"])
    arena_height_cm = float(row["arena_height_cm"])

    left_px = float(row["left_px"])
    right_px = float(row["right_px"])
    top_px = float(row["top_px"])
    bottom_px = float(row["bottom_px"])

    if arena_width_cm <= 0 or arena_height_cm <= 0:
        raise ValueError("Arena dimensions must be greater than zero.")

    if right_px <= left_px or bottom_px <= top_px:
        raise ValueError("Arena pixel boundaries are invalid.")

    x_px_per_cm = (right_px - left_px) / arena_width_cm
    y_px_per_cm = (bottom_px - top_px) / arena_height_cm

    return {
        "arena_width_cm": arena_width_cm,
        "arena_height_cm": arena_height_cm,
        "left_px": left_px,
        "right_px": right_px,
        "top_px": top_px,
        "bottom_px": bottom_px,
        "x_px_per_cm": x_px_per_cm,
        "y_px_per_cm": y_px_per_cm,
    }


def load_reader_geometry(
    path: Path,
    calibration: dict[str, float],
) -> pd.DataFrame:
    readers = pd.read_csv(path)

    require_columns(
        readers,
        [
            "reader",
            "x_cm_from_left",
            "y_cm_from_top",
            "width_x_cm",
            "length_y_cm",
        ],
        "Reader configuration",
    )

    readers = readers.copy()

    if readers["reader"].duplicated().any():
        duplicates = readers.loc[
            readers["reader"].duplicated(keep=False),
            "reader",
        ].tolist()

        raise ValueError(
            f"Reader configuration contains duplicate readers: {duplicates}"
        )

    numeric_columns = [
        "x_cm_from_left",
        "y_cm_from_top",
        "width_x_cm",
        "length_y_cm",
    ]

    for column in numeric_columns:
        readers[column] = pd.to_numeric(
            readers[column],
            errors="raise",
        )

    readers["reader_centre_x_px"] = (
        calibration["left_px"]
        + readers["x_cm_from_left"] * calibration["x_px_per_cm"]
    )

    readers["reader_centre_y_px"] = (
        calibration["top_px"]
        + readers["y_cm_from_top"] * calibration["y_px_per_cm"]
    )

    readers["reader_width_px"] = (
        readers["width_x_cm"] * calibration["x_px_per_cm"]
    )

    readers["reader_length_px"] = (
        readers["length_y_cm"] * calibration["y_px_per_cm"]
    )

    readers["reader_left_px"] = (
        readers["reader_centre_x_px"]
        - readers["reader_width_px"] / 2.0
    )

    readers["reader_right_px"] = (
        readers["reader_centre_x_px"]
        + readers["reader_width_px"] / 2.0
    )

    readers["reader_top_px"] = (
        readers["reader_centre_y_px"]
        - readers["reader_length_px"] / 2.0
    )

    readers["reader_bottom_px"] = (
        readers["reader_centre_y_px"]
        + readers["reader_length_px"] / 2.0
    )

    return readers


def prepare_sleap(path: Path) -> pd.DataFrame:
    sleap = pd.read_csv(path)

    require_columns(
        sleap,
        [
            "frame_idx",
            "instance_index",
            "track_id",
            "instance_score",
            "tracking_score",
            "n_visible",
            "torso_centre_x",
            "torso_centre_y",
            "torso_centre_score",
            "torso_centre_visible",
            "absolute_timestamp",
        ],
        "Timestamped SLEAP table",
    )

    sleap = sleap.copy()

    numeric_columns = [
        "frame_idx",
        "instance_index",
        "instance_score",
        "tracking_score",
        "n_visible",
        "torso_centre_x",
        "torso_centre_y",
        "torso_centre_score",
    ]

    for column in numeric_columns:
        sleap[column] = pd.to_numeric(
            sleap[column],
            errors="coerce",
        )

    sleap["frame_idx"] = sleap["frame_idx"].astype("Int64")

    # A torso coordinate is usable only when x and y are finite.
    sleap["torso_position_usable"] = (
        sleap["torso_centre_x"].notna()
        & sleap["torso_centre_y"].notna()
        & np.isfinite(sleap["torso_centre_x"])
        & np.isfinite(sleap["torso_centre_y"])
    )

    return sleap


def prepare_rfid(path: Path) -> pd.DataFrame:
    rfid = pd.read_csv(path)

    require_columns(
        rfid,
        [
            "raw_row_number",
            "mouse_id",
            "tag_id",
            "reader",
            "duration_ms",
            "rfid_records",
            "matching_timestamp",
            "nearest_frame_idx",
            "nearest_frame_timestamp",
            "video_elapsed_seconds",
            "frame_time_difference_ms",
        ],
        "Aligned RFID table",
    )

    rfid = rfid.copy()

    rfid["nearest_frame_idx"] = pd.to_numeric(
        rfid["nearest_frame_idx"],
        errors="raise",
    ).astype(int)

    rfid["mouse_id"] = rfid["mouse_id"].astype(str)

    return rfid


def point_inside_reader(
    x_px: float,
    y_px: float,
    reader: pd.Series,
) -> bool:
    return bool(
        reader["reader_left_px"] <= x_px <= reader["reader_right_px"]
        and reader["reader_top_px"] <= y_px <= reader["reader_bottom_px"]
    )


def calculate_candidate_table(
    frame_detections: pd.DataFrame,
    reader: pd.Series,
    calibration: dict[str, float],
) -> pd.DataFrame:
    candidates = frame_detections.loc[
        frame_detections["torso_position_usable"]
    ].copy()

    if candidates.empty:
        return candidates

    candidates["reader_delta_x_px"] = (
        candidates["torso_centre_x"]
        - reader["reader_centre_x_px"]
    )

    candidates["reader_delta_y_px"] = (
        candidates["torso_centre_y"]
        - reader["reader_centre_y_px"]
    )

    candidates["reader_distance_px"] = np.sqrt(
        candidates["reader_delta_x_px"] ** 2
        + candidates["reader_delta_y_px"] ** 2
    )

    candidates["reader_delta_x_cm"] = (
        candidates["reader_delta_x_px"]
        / calibration["x_px_per_cm"]
    )

    candidates["reader_delta_y_cm"] = (
        candidates["reader_delta_y_px"]
        / calibration["y_px_per_cm"]
    )

    candidates["reader_distance_cm"] = np.sqrt(
        candidates["reader_delta_x_cm"] ** 2
        + candidates["reader_delta_y_cm"] ** 2
    )

    candidates["inside_reader_rectangle"] = candidates.apply(
        lambda row: point_inside_reader(
            float(row["torso_centre_x"]),
            float(row["torso_centre_y"]),
            reader,
        ),
        axis=1,
    )

    candidates = candidates.sort_values(
        by=[
            "reader_distance_cm",
            "torso_centre_score",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    candidates["candidate_rank"] = np.arange(
        1,
        len(candidates) + 1,
    )

    return candidates


def candidate_value(
    candidates: pd.DataFrame,
    rank_index: int,
    column: str,
):
    if len(candidates) <= rank_index:
        return np.nan

    return candidates.iloc[rank_index][column]


def classify_anchor(
    candidate_count: int,
    nearest_distance_cm: float,
    margin_cm: float,
    ambiguity_margin_cm: float,
    far_distance_cm: float,
) -> tuple[str, bool, bool]:
    if candidate_count == 0:
        return "no_usable_candidate", False, False

    is_far = bool(
        pd.notna(nearest_distance_cm)
        and nearest_distance_cm > far_distance_cm
    )

    is_ambiguous = bool(
        candidate_count >= 2
        and pd.notna(margin_cm)
        and margin_cm < ambiguity_margin_cm
    )

    if is_far and is_ambiguous:
        return "provisional_far_and_ambiguous", True, True

    if is_far:
        return "provisional_far", True, False

    if is_ambiguous:
        return "provisional_ambiguous", False, True

    return "provisional_nearest", False, False


def assign_anchors(
    sleap: pd.DataFrame,
    rfid: pd.DataFrame,
    readers: pd.DataFrame,
    calibration: dict[str, float],
    ambiguity_margin_cm: float,
    far_distance_cm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reader_lookup = readers.set_index("reader", drop=False)

    sleap_by_frame = {
        int(frame_idx): frame.copy()
        for frame_idx, frame in sleap.groupby("frame_idx", sort=False)
        if pd.notna(frame_idx)
    }

    anchor_rows: list[dict] = []
    all_candidate_rows: list[dict] = []

    for event_index, event in rfid.reset_index(drop=True).iterrows():
        reader_id = str(event["reader"])
        frame_idx = int(event["nearest_frame_idx"])

        if reader_id not in reader_lookup.index:
            raise ValueError(
                f"RFID event references unknown reader: {reader_id}"
            )

        reader = reader_lookup.loc[reader_id]

        frame_detections = sleap_by_frame.get(
            frame_idx,
            sleap.iloc[0:0].copy(),
        )

        candidates = calculate_candidate_table(
            frame_detections,
            reader,
            calibration,
        )

        for _, candidate in candidates.iterrows():
            all_candidate_rows.append(
                {
                    "rfid_event_index": event_index,
                    "raw_row_number": event["raw_row_number"],
                    "mouse_id": event["mouse_id"],
                    "tag_id": event["tag_id"],
                    "reader": reader_id,
                    "nearest_frame_idx": frame_idx,
                    "matching_timestamp": event["matching_timestamp"],
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "candidate_track_id": candidate["track_id"],
                    "candidate_instance_index": candidate["instance_index"],
                    "candidate_torso_x_px": candidate["torso_centre_x"],
                    "candidate_torso_y_px": candidate["torso_centre_y"],
                    "candidate_torso_score": candidate[
                        "torso_centre_score"
                    ],
                    "candidate_instance_score": candidate[
                        "instance_score"
                    ],
                    "candidate_tracking_score": candidate[
                        "tracking_score"
                    ],
                    "candidate_n_visible": candidate["n_visible"],
                    "candidate_distance_px": candidate[
                        "reader_distance_px"
                    ],
                    "candidate_distance_cm": candidate[
                        "reader_distance_cm"
                    ],
                    "candidate_inside_reader_rectangle": candidate[
                        "inside_reader_rectangle"
                    ],
                }
            )

        candidate_count = len(candidates)

        nearest_distance_cm = candidate_value(
            candidates,
            0,
            "reader_distance_cm",
        )

        second_distance_cm = candidate_value(
            candidates,
            1,
            "reader_distance_cm",
        )

        if (
            pd.notna(nearest_distance_cm)
            and pd.notna(second_distance_cm)
        ):
            margin_cm = (
                float(second_distance_cm)
                - float(nearest_distance_cm)
            )
        else:
            margin_cm = np.nan

        status, far_flag, ambiguity_flag = classify_anchor(
            candidate_count=candidate_count,
            nearest_distance_cm=nearest_distance_cm,
            margin_cm=margin_cm,
            ambiguity_margin_cm=ambiguity_margin_cm,
            far_distance_cm=far_distance_cm,
        )

        anchor_rows.append(
            {
                "rfid_event_index": event_index,
                "raw_row_number": event["raw_row_number"],
                "mouse_id": event["mouse_id"],
                "tag_id": event["tag_id"],
                "reader": reader_id,
                "event_start_timestamp": event.get(
                    "event_start_timestamp",
                    np.nan,
                ),
                "event_end_timestamp": event.get(
                    "event_end_timestamp",
                    np.nan,
                ),
                "duration_ms": event["duration_ms"],
                "rfid_records": event["rfid_records"],
                "matching_timestamp": event["matching_timestamp"],
                "nearest_frame_idx": frame_idx,
                "nearest_frame_timestamp": event[
                    "nearest_frame_timestamp"
                ],
                "video_elapsed_seconds": event[
                    "video_elapsed_seconds"
                ],
                "frame_time_difference_ms": event[
                    "frame_time_difference_ms"
                ],
                "reader_centre_x_px": reader[
                    "reader_centre_x_px"
                ],
                "reader_centre_y_px": reader[
                    "reader_centre_y_px"
                ],
                "reader_width_cm": reader["width_x_cm"],
                "reader_length_cm": reader["length_y_cm"],
                "sleap_instances_in_frame": len(frame_detections),
                "usable_torso_candidates": candidate_count,
                "assigned_track_id": candidate_value(
                    candidates,
                    0,
                    "track_id",
                ),
                "assigned_instance_index": candidate_value(
                    candidates,
                    0,
                    "instance_index",
                ),
                "assigned_torso_x_px": candidate_value(
                    candidates,
                    0,
                    "torso_centre_x",
                ),
                "assigned_torso_y_px": candidate_value(
                    candidates,
                    0,
                    "torso_centre_y",
                ),
                "assigned_torso_score": candidate_value(
                    candidates,
                    0,
                    "torso_centre_score",
                ),
                "assigned_instance_score": candidate_value(
                    candidates,
                    0,
                    "instance_score",
                ),
                "assigned_tracking_score": candidate_value(
                    candidates,
                    0,
                    "tracking_score",
                ),
                "assigned_n_visible": candidate_value(
                    candidates,
                    0,
                    "n_visible",
                ),
                "nearest_distance_px": candidate_value(
                    candidates,
                    0,
                    "reader_distance_px",
                ),
                "nearest_distance_cm": nearest_distance_cm,
                "nearest_inside_reader_rectangle": candidate_value(
                    candidates,
                    0,
                    "inside_reader_rectangle",
                ),
                "second_track_id": candidate_value(
                    candidates,
                    1,
                    "track_id",
                ),
                "second_distance_px": candidate_value(
                    candidates,
                    1,
                    "reader_distance_px",
                ),
                "second_distance_cm": second_distance_cm,
                "nearest_second_margin_cm": margin_cm,
                "assignment_source": (
                    "direct_rfid_provisional"
                    if candidate_count > 0
                    else "unassigned"
                ),
                "assignment_status": status,
                "far_distance_flag": far_flag,
                "ambiguity_flag": ambiguity_flag,
                "ambiguity_margin_threshold_cm": (
                    ambiguity_margin_cm
                ),
                "far_distance_threshold_cm": far_distance_cm,
            }
        )

    anchors = pd.DataFrame(anchor_rows)
    candidates = pd.DataFrame(all_candidate_rows)

    return anchors, candidates


def build_summary(
    anchors: pd.DataFrame,
    sleap: pd.DataFrame,
) -> pd.DataFrame:
    total = len(anchors)
    assigned = int(anchors["assigned_track_id"].notna().sum())
    no_candidate = int(
        (anchors["usable_torso_candidates"] == 0).sum()
    )

    inside = int(
        anchors["nearest_inside_reader_rectangle"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    ambiguous = int(
        anchors["ambiguity_flag"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    far = int(
        anchors["far_distance_flag"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    distance = anchors["nearest_distance_cm"].dropna()
    margin = anchors["nearest_second_margin_cm"].dropna()

    summary_rows = [
        ("rfid_events_total", total),
        ("rfid_events_assigned_provisionally", assigned),
        ("rfid_events_without_usable_candidate", no_candidate),
        (
            "provisional_assignment_percent",
            100.0 * assigned / total if total else np.nan,
        ),
        (
            "nearest_track_inside_reader_rectangle_count",
            inside,
        ),
        (
            "nearest_track_inside_reader_rectangle_percent",
            100.0 * inside / assigned if assigned else np.nan,
        ),
        ("ambiguous_event_count", ambiguous),
        (
            "ambiguous_event_percent_of_assigned",
            100.0 * ambiguous / assigned if assigned else np.nan,
        ),
        ("far_event_count", far),
        (
            "far_event_percent_of_assigned",
            100.0 * far / assigned if assigned else np.nan,
        ),
        (
            "nearest_distance_cm_median",
            distance.median() if not distance.empty else np.nan,
        ),
        (
            "nearest_distance_cm_mean",
            distance.mean() if not distance.empty else np.nan,
        ),
        (
            "nearest_distance_cm_standard_deviation",
            distance.std(ddof=1) if len(distance) > 1 else np.nan,
        ),
        (
            "nearest_distance_cm_minimum",
            distance.min() if not distance.empty else np.nan,
        ),
        (
            "nearest_distance_cm_maximum",
            distance.max() if not distance.empty else np.nan,
        ),
        (
            "nearest_second_margin_cm_median",
            margin.median() if not margin.empty else np.nan,
        ),
        (
            "unique_sleap_track_fragments",
            sleap["track_id"].nunique(dropna=True),
        ),
        (
            "video_frames_with_sleap_detections",
            sleap["frame_idx"].nunique(dropna=True),
        ),
    ]

    return pd.DataFrame(
        summary_rows,
        columns=["metric", "value"],
    )


def build_group_summary(
    anchors: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict] = []

    for group_value, group in anchors.groupby(
        group_column,
        dropna=False,
    ):
        assigned = group["assigned_track_id"].notna()
        inside = (
            group["nearest_inside_reader_rectangle"]
            .fillna(False)
            .astype(bool)
        )
        ambiguous = (
            group["ambiguity_flag"]
            .fillna(False)
            .astype(bool)
        )
        far = (
            group["far_distance_flag"]
            .fillna(False)
            .astype(bool)
        )

        distances = group["nearest_distance_cm"].dropna()

        rows.append(
            {
                group_column: group_value,
                "rfid_events": len(group),
                "assigned_events": int(assigned.sum()),
                "assigned_percent": (
                    100.0 * assigned.mean()
                    if len(group)
                    else np.nan
                ),
                "inside_reader_count": int(inside.sum()),
                "inside_reader_percent_of_assigned": (
                    100.0 * inside.sum() / assigned.sum()
                    if assigned.sum()
                    else np.nan
                ),
                "ambiguous_count": int(ambiguous.sum()),
                "far_count": int(far.sum()),
                "median_nearest_distance_cm": (
                    distances.median()
                    if not distances.empty
                    else np.nan
                ),
                "mean_nearest_distance_cm": (
                    distances.mean()
                    if not distances.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def save_outputs(
    anchors: pd.DataFrame,
    candidates: pd.DataFrame,
    summary: pd.DataFrame,
    by_reader: pd.DataFrame,
    by_mouse: pd.DataFrame,
    output_dir: Path,
    video_id: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "anchors": (
            output_dir
            / f"{video_id}_rfid_anchor_assignments.csv"
        ),
        "candidates": (
            output_dir
            / f"{video_id}_rfid_anchor_candidates.csv"
        ),
        "summary": (
            output_dir
            / f"{video_id}_rfid_anchor_summary.csv"
        ),
        "reader_summary": (
            output_dir
            / f"{video_id}_rfid_anchor_summary_by_reader.csv"
        ),
        "mouse_summary": (
            output_dir
            / f"{video_id}_rfid_anchor_summary_by_mouse.csv"
        ),
    }

    anchors.to_csv(paths["anchors"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    summary.to_csv(paths["summary"], index=False)
    by_reader.to_csv(paths["reader_summary"], index=False)
    by_mouse.to_csv(paths["mouse_summary"], index=False)

    return paths


def main() -> None:
    args = parse_args()

    require_file(args.sleap, "Timestamped SLEAP table")
    require_file(args.rfid, "Aligned RFID table")
    require_file(args.reader_cm, "Reader configuration")
    require_file(args.calibration, "Arena calibration")

    calibration = load_calibration(args.calibration)

    readers = load_reader_geometry(
        args.reader_cm,
        calibration,
    )

    sleap = prepare_sleap(args.sleap)
    rfid = prepare_rfid(args.rfid)

    unknown_readers = sorted(
        set(rfid["reader"].astype(str))
        - set(readers["reader"].astype(str))
    )

    if unknown_readers:
        raise ValueError(
            "The aligned RFID file contains readers absent from "
            f"the reader configuration: {unknown_readers}"
        )

    anchors, candidates = assign_anchors(
        sleap=sleap,
        rfid=rfid,
        readers=readers,
        calibration=calibration,
        ambiguity_margin_cm=args.ambiguity_margin_cm,
        far_distance_cm=args.far_distance_cm,
    )

    summary = build_summary(
        anchors=anchors,
        sleap=sleap,
    )

    by_reader = build_group_summary(
        anchors,
        "reader",
    )

    by_mouse = build_group_summary(
        anchors,
        "mouse_id",
    )

    paths = save_outputs(
        anchors=anchors,
        candidates=candidates,
        summary=summary,
        by_reader=by_reader,
        by_mouse=by_mouse,
        output_dir=args.output_dir,
        video_id=args.video_id,
    )

    print()
    print("Stage 11A complete")
    print("==================")
    print(f"Video ID: {args.video_id}")
    print(f"RFID events evaluated: {len(anchors)}")
    print(
        "Events assigned provisionally: "
        f"{anchors['assigned_track_id'].notna().sum()}"
    )
    print(
        "Events with no usable torso candidate: "
        f"{(anchors['usable_torso_candidates'] == 0).sum()}"
    )
    print(
        "Ambiguous events: "
        f"{anchors['ambiguity_flag'].sum()}"
    )
    print(
        "Spatially distant events: "
        f"{anchors['far_distance_flag'].sum()}"
    )

    distances = anchors["nearest_distance_cm"].dropna()

    if not distances.empty:
        print(
            "Median nearest-track distance: "
            f"{distances.median():.3f} cm"
        )
        print(
            "Mean nearest-track distance: "
            f"{distances.mean():.3f} cm"
        )

    print()
    print("Outputs:")

    for label, path in paths.items():
        print(f"  {label}: {path}")

    print()
    print("Summary statistics:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
