#!/usr/bin/env python3

"""
Create corrected RFID reader overlays in Basler image coordinates.

Outputs:
1. Corrected reader geometry overlay.
2. RFID anchor validation overlay containing all provisional assignments.

The physical RFID layout is transformed by 180 degrees before use in
the image coordinate system. This script expects the transformed reader
configuration produced during Stage 11A validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


DEFAULT_VIDEO_ID = "407405spareat1313"

DEFAULT_CALIBRATION = Path(
    "config/arena_calibration_407405spareat1313.csv"
)

DEFAULT_READERS = Path(
    "config/rfid_reader_positions_cm_basler_180.csv"
)

DEFAULT_ANCHORS = Path(
    "outputs/development_v1/identity/"
    "407405spareat1313_rfid_anchor_assignments.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/development_v1/identity/figures"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-id",
        default=DEFAULT_VIDEO_ID,
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )

    parser.add_argument(
        "--readers",
        type=Path,
        default=DEFAULT_READERS,
    )

    parser.add_argument(
        "--anchors",
        type=Path,
        default=DEFAULT_ANCHORS,
    )

    parser.add_argument(
        "--background",
        type=Path,
        default=None,
        help=(
            "Optional arena image or extracted video frame. "
            "If omitted, a blank image-coordinate canvas is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def require_columns(
    table: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = [
        column for column in columns
        if column not in table.columns
    ]

    if missing:
        raise ValueError(
            f"{name} missing required columns: {missing}"
        )


def load_geometry(
    calibration_path: Path,
    readers_path: Path,
) -> tuple[pd.Series, pd.DataFrame]:
    calibration = pd.read_csv(calibration_path)

    if len(calibration) != 1:
        raise ValueError(
            "Calibration table must contain exactly one row."
        )

    cal = calibration.iloc[0]

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
        "Calibration",
    )

    readers = pd.read_csv(readers_path)

    require_columns(
        readers,
        [
            "reader",
            "x_cm_from_left",
            "y_cm_from_top",
            "width_x_cm",
            "length_y_cm",
        ],
        "Reader geometry",
    )

    x_px_per_cm = (
        float(cal["right_px"]) - float(cal["left_px"])
    ) / float(cal["arena_width_cm"])

    y_px_per_cm = (
        float(cal["bottom_px"]) - float(cal["top_px"])
    ) / float(cal["arena_height_cm"])

    readers = readers.copy()

    readers["centre_x_px"] = (
        float(cal["left_px"])
        + readers["x_cm_from_left"] * x_px_per_cm
    )

    readers["centre_y_px"] = (
        float(cal["top_px"])
        + readers["y_cm_from_top"] * y_px_per_cm
    )

    readers["width_px"] = (
        readers["width_x_cm"] * x_px_per_cm
    )

    readers["height_px"] = (
        readers["length_y_cm"] * y_px_per_cm
    )

    readers["left_px"] = (
        readers["centre_x_px"] - readers["width_px"] / 2
    )

    readers["top_px"] = (
        readers["centre_y_px"] - readers["height_px"] / 2
    )

    return cal, readers


def load_background(
    path: Path | None,
) -> np.ndarray | None:
    if path is None:
        return None

    if not path.exists():
        raise FileNotFoundError(
            f"Background image not found: {path}"
        )

    return plt.imread(path)


def configure_axes(
    ax,
    cal: pd.Series,
    background: np.ndarray | None,
) -> None:
    if background is not None:
        ax.imshow(background)

    ax.set_xlim(
        float(cal["left_px"]),
        float(cal["right_px"]),
    )

    # Image y coordinates increase downward.
    ax.set_ylim(
        float(cal["bottom_px"]),
        float(cal["top_px"]),
    )

    ax.set_aspect("equal")

    ax.set_xlabel("Basler image x coordinate (pixels)")
    ax.set_ylabel("Basler image y coordinate (pixels)")

    arena = Rectangle(
        (
            float(cal["left_px"]),
            float(cal["top_px"]),
        ),
        float(cal["right_px"]) - float(cal["left_px"]),
        float(cal["bottom_px"]) - float(cal["top_px"]),
        fill=False,
        linewidth=2,
        linestyle="--",
        edgecolor="black",
        label="Calibrated arena boundary",
    )

    ax.add_patch(arena)


def draw_readers(
    ax,
    readers: pd.DataFrame,
) -> None:
    for _, reader in readers.iterrows():
        rectangle = Rectangle(
            (
                reader["left_px"],
                reader["top_px"],
            ),
            reader["width_px"],
            reader["height_px"],
            fill=False,
            linewidth=2.5,
            edgecolor="red",
        )

        ax.add_patch(rectangle)

        ax.scatter(
            reader["centre_x_px"],
            reader["centre_y_px"],
            marker="+",
            s=120,
            linewidth=2,
            color="red",
        )

        ax.text(
            reader["centre_x_px"],
            reader["centre_y_px"] - reader["height_px"] / 2 - 8,
            str(reader["reader"]),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=11,
            fontweight="bold",
            color="red",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.5,
            },
        )


def save_reader_overlay(
    cal: pd.Series,
    readers: pd.DataFrame,
    background: np.ndarray | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))

    configure_axes(ax, cal, background)
    draw_readers(ax, readers)

    ax.set_title(
        "Corrected RFID reader geometry\n"
        "180° transformation into Basler image coordinates"
    )

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_anchor_overlay(
    cal: pd.Series,
    readers: pd.DataFrame,
    anchors: pd.DataFrame,
    background: np.ndarray | None,
    output_path: Path,
) -> None:
    require_columns(
        anchors,
        [
            "mouse_id",
            "reader",
            "assigned_torso_x_px",
            "assigned_torso_y_px",
            "nearest_distance_cm",
            "far_distance_flag",
            "ambiguity_flag",
        ],
        "Anchor assignments",
    )

    anchors = anchors.loc[
        anchors["assigned_torso_x_px"].notna()
        & anchors["assigned_torso_y_px"].notna()
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 9))

    configure_axes(ax, cal, background)
    draw_readers(ax, readers)

    mouse_ids = sorted(
        anchors["mouse_id"].astype(str).unique()
    )

    markers = ["o", "^", "s", "D", "v"]

    for index, mouse_id in enumerate(mouse_ids):
        group = anchors.loc[
            anchors["mouse_id"].astype(str) == mouse_id
        ]

        ax.scatter(
            group["assigned_torso_x_px"],
            group["assigned_torso_y_px"],
            s=30,
            alpha=0.55,
            marker=markers[index % len(markers)],
            label=f"Mouse {mouse_id}",
        )

    far = anchors.loc[
        anchors["far_distance_flag"]
        .fillna(False)
        .astype(bool)
    ]

    if not far.empty:
        ax.scatter(
            far["assigned_torso_x_px"],
            far["assigned_torso_y_px"],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Distance > 8 cm",
        )

    ambiguous = anchors.loc[
        anchors["ambiguity_flag"]
        .fillna(False)
        .astype(bool)
    ]

    if not ambiguous.empty:
        ax.scatter(
            ambiguous["assigned_torso_x_px"],
            ambiguous["assigned_torso_y_px"],
            s=150,
            marker="x",
            linewidths=2,
            color="black",
            label="Ambiguous event",
        )

    median_distance = anchors["nearest_distance_cm"].median()

    ax.set_title(
        "Corrected RFID–SLEAP anchor validation\n"
        f"{len(anchors)} RFID events; "
        f"median distance = {median_distance:.2f} cm"
    )

    ax.legend(
        loc="upper right",
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_reader_summary(
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    return (
        anchors.groupby("reader")
        .agg(
            events=("reader", "size"),
            median_distance_cm=(
                "nearest_distance_cm",
                "median",
            ),
            mean_distance_cm=(
                "nearest_distance_cm",
                "mean",
            ),
            maximum_distance_cm=(
                "nearest_distance_cm",
                "max",
            ),
            far_events=(
                "far_distance_flag",
                "sum",
            ),
            ambiguous_events=(
                "ambiguity_flag",
                "sum",
            ),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()

    for path, description in [
        (args.calibration, "Calibration"),
        (args.readers, "Reader geometry"),
        (args.anchors, "Anchor assignments"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cal, readers = load_geometry(
        args.calibration,
        args.readers,
    )

    anchors = pd.read_csv(args.anchors)
    background = load_background(args.background)

    reader_output = (
        args.output_dir
        / f"{args.video_id}_corrected_reader_overlay.png"
    )

    anchor_output = (
        args.output_dir
        / f"{args.video_id}_corrected_rfid_anchor_overlay.png"
    )

    summary_output = (
        args.output_dir
        / f"{args.video_id}_overlay_summary_by_reader.csv"
    )

    save_reader_overlay(
        cal,
        readers,
        background,
        reader_output,
    )

    save_anchor_overlay(
        cal,
        readers,
        anchors,
        background,
        anchor_output,
    )

    summary = build_reader_summary(anchors)
    summary.to_csv(summary_output, index=False)

    print()
    print("Corrected overlay generation complete")
    print("=====================================")
    print(f"Reader overlay: {reader_output}")
    print(f"Anchor overlay: {anchor_output}")
    print(f"Reader summary: {summary_output}")

    print("\nAnchor summary by reader:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
