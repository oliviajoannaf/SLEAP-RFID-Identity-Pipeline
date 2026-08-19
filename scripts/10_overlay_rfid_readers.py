#!/usr/bin/env python3

from pathlib import Path
import argparse

import cv2
import pandas as pd


def cm_to_pixel(
    x_cm: float,
    y_cm: float,
    left_px: float,
    right_px: float,
    top_px: float,
    bottom_px: float,
    arena_width_cm: float,
    arena_height_cm: float,
) -> tuple[float, float]:
    """Convert arena coordinates in centimetres to image pixels."""
    x_px = left_px + (x_cm / arena_width_cm) * (right_px - left_px)
    y_px = top_px + (y_cm / arena_height_cm) * (bottom_px - top_px)
    return x_px, y_px


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RFID reader positions from centimetres to pixels "
        "and draw them on an arena calibration frame."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--arena-calibration", required=True, type=Path)
    parser.add_argument("--reader-config", required=True, type=Path)
    parser.add_argument("--output-image", required=True, type=Path)
    parser.add_argument("--output-table", required=True, type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    calibration_df = pd.read_csv(args.arena_calibration)
    if len(calibration_df) != 1:
        raise ValueError("Arena calibration file must contain exactly one row.")

    calibration = calibration_df.iloc[0]

    left_px = float(calibration["left_px"])
    right_px = float(calibration["right_px"])
    top_px = float(calibration["top_px"])
    bottom_px = float(calibration["bottom_px"])
    arena_width_cm = float(calibration["arena_width_cm"])
    arena_height_cm = float(calibration["arena_height_cm"])

    readers = pd.read_csv(args.reader_config)

    output_rows = []

    # Draw the calibrated 51 × 51 cm arena boundary.
    cv2.rectangle(
        image,
        (round(left_px), round(top_px)),
        (round(right_px), round(bottom_px)),
        (255, 255, 255),
        2,
    )

    for _, reader in readers.iterrows():
        reader_name = str(reader["reader"])

        centre_x_cm = float(reader["x_cm_from_left"])
        centre_y_cm = float(reader["y_cm_from_top"])
        width_cm = float(reader["width_x_cm"])
        height_cm = float(reader["length_y_cm"])

        x_min_cm = centre_x_cm - width_cm / 2
        x_max_cm = centre_x_cm + width_cm / 2
        y_min_cm = centre_y_cm - height_cm / 2
        y_max_cm = centre_y_cm + height_cm / 2

        centre_x_px, centre_y_px = cm_to_pixel(
            centre_x_cm,
            centre_y_cm,
            left_px,
            right_px,
            top_px,
            bottom_px,
            arena_width_cm,
            arena_height_cm,
        )

        x_min_px, y_min_px = cm_to_pixel(
            x_min_cm,
            y_min_cm,
            left_px,
            right_px,
            top_px,
            bottom_px,
            arena_width_cm,
            arena_height_cm,
        )

        x_max_px, y_max_px = cm_to_pixel(
            x_max_cm,
            y_max_cm,
            left_px,
            right_px,
            top_px,
            bottom_px,
            arena_width_cm,
            arena_height_cm,
        )

        cv2.rectangle(
            image,
            (round(x_min_px), round(y_min_px)),
            (round(x_max_px), round(y_max_px)),
            (0, 0, 255),
            3,
        )

        cv2.circle(
            image,
            (round(centre_x_px), round(centre_y_px)),
            5,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            reader_name,
            (round(x_min_px), round(y_min_px) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        output_rows.append(
            {
                "reader": reader_name,
                "centre_x_cm": centre_x_cm,
                "centre_y_cm": centre_y_cm,
                "x_min_cm": x_min_cm,
                "x_max_cm": x_max_cm,
                "y_min_cm": y_min_cm,
                "y_max_cm": y_max_cm,
                "centre_x_px": centre_x_px,
                "centre_y_px": centre_y_px,
                "x_min_px": x_min_px,
                "x_max_px": x_max_px,
                "y_min_px": y_min_px,
                "y_max_px": y_max_px,
                "calibration_status": calibration["status"],
            }
        )

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(args.output_image), image):
        raise RuntimeError(f"Could not save overlay: {args.output_image}")

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output_table, index=False)

    x_scale = (right_px - left_px) / arena_width_cm
    y_scale = (bottom_px - top_px) / arena_height_cm

    print("Stage 10 complete")
    print(f"Arena x scale: {x_scale:.4f} px/cm")
    print(f"Arena y scale: {y_scale:.4f} px/cm")
    print(f"Reader table: {args.output_table}")
    print(f"Overlay image: {args.output_image}")
    print()
    print(output_df[
        ["reader", "centre_x_px", "centre_y_px",
         "x_min_px", "x_max_px", "y_min_px", "y_max_px"]
    ].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
