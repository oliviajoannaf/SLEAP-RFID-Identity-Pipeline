from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


INPUT = Path("outputs/experimental_v2/final_identity_threshold_075")
OUTPUT = Path("outputs/experimental_v2/behaviour_complete")
OUTPUT.mkdir(parents=True, exist_ok=True)

FPS = 30.0
PIXELS_PER_CM = 8.75

# QC threshold for implausible frame-to-frame displacement
MAX_STEP_CM = 5.0

# Exploratory social-distance threshold
PROXIMITY_THRESHOLD_CM = 5.0


all_animal_results = []
all_pair_results = []


for file in sorted(INPUT.glob("*_identity_propagated.csv")):

    recording = file.name.replace("_identity_propagated.csv", "")
    print(f"\nProcessing: {recording}")

    df = pd.read_csv(file)

    # Retain only rows with an RFID-resolved identity
    df = df[df["identity_is_known"] == True].copy()
    df = df.dropna(subset=["mouse_id", "centroid_x", "centroid_y"])

    df["mouse_id"] = df["mouse_id"].astype(int)

    # Where duplicate detections exist for one mouse in one frame,
    # retain the highest-scoring detection
    if "instance_score" in df.columns:
        df = (
            df.sort_values("instance_score", ascending=False)
              .drop_duplicates(["frame_idx", "mouse_id"])
        )
    else:
        df = df.drop_duplicates(["frame_idx", "mouse_id"])

    df["x_cm"] = df["centroid_x"] / PIXELS_PER_CM
    df["y_cm"] = df["centroid_y"] / PIXELS_PER_CM

    # Estimate arena limits from the fixed 800 x 800 pixel recording
    arena_width_cm = 800 / PIXELS_PER_CM
    arena_height_cm = 800 / PIXELS_PER_CM

    # Centre defined as central 50% of arena width and height
    centre_x_min = arena_width_cm * 0.25
    centre_x_max = arena_width_cm * 0.75
    centre_y_min = arena_height_cm * 0.25
    centre_y_max = arena_height_cm * 0.75

    df["in_centre"] = (
        df["x_cm"].between(centre_x_min, centre_x_max)
        & df["y_cm"].between(centre_y_min, centre_y_max)
    )

    animal_results = []

    for mouse, g in df.groupby("mouse_id"):

        g = g.sort_values("frame_idx").copy()

        frame_gap = g["frame_idx"].diff()
        dx = g["x_cm"].diff()
        dy = g["y_cm"].diff()

        step = np.sqrt(dx**2 + dy**2)

        # Only use movement between genuinely consecutive video frames
        valid_step = (frame_gap == 1) & (step <= MAX_STEP_CM)
        filtered_step = step.where(valid_step)

        distance_cm = filtered_step.sum(skipna=True)

        valid_movement_frames = int(valid_step.sum())
        movement_duration_s = valid_movement_frames / FPS

        mean_speed_cm_s = (
            distance_cm / movement_duration_s
            if movement_duration_s > 0
            else np.nan
        )

        observed_frames = len(g)
        observed_time_s = observed_frames / FPS

        centre_frames = int(g["in_centre"].sum())
        centre_time_s = centre_frames / FPS
        centre_occupancy_percent = (
            100 * centre_frames / observed_frames
            if observed_frames > 0
            else np.nan
        )

        result = {
            "recording": recording,
            "mouse_id": mouse,
            "identity_resolved_frames": observed_frames,
            "identity_resolved_time_s": observed_time_s,
            "valid_movement_frames": valid_movement_frames,
            "distance_cm": distance_cm,
            "mean_speed_cm_s": mean_speed_cm_s,
            "centre_frames": centre_frames,
            "centre_time_s": centre_time_s,
            "centre_occupancy_percent": centre_occupancy_percent,
        }

        animal_results.append(result)
        all_animal_results.append(result)

    animal_out = pd.DataFrame(animal_results)

    animal_out.to_csv(
        OUTPUT / f"{recording}_animal_behaviour.csv",
        index=False
    )

    print("\nAnimal behaviour:")
    print(
        animal_out[
            [
                "mouse_id",
                "identity_resolved_frames",
                "distance_cm",
                "mean_speed_cm_s",
                "centre_occupancy_percent",
            ]
        ].round(2)
    )

    # ---------------------------------------------------------
    # Pairwise proximity
    # ---------------------------------------------------------

    pair_results = []
    mice = sorted(df["mouse_id"].unique())

    for mouse_a, mouse_b in combinations(mice, 2):

        a = (
            df[df["mouse_id"] == mouse_a]
            [["frame_idx", "x_cm", "y_cm"]]
            .rename(columns={"x_cm": "x_a", "y_cm": "y_a"})
        )

        b = (
            df[df["mouse_id"] == mouse_b]
            [["frame_idx", "x_cm", "y_cm"]]
            .rename(columns={"x_cm": "x_b", "y_cm": "y_b"})
        )

        pair = a.merge(b, on="frame_idx", how="inner")

        pair["distance_cm"] = np.sqrt(
            (pair["x_a"] - pair["x_b"]) ** 2
            + (pair["y_a"] - pair["y_b"]) ** 2
        )

        pair["in_proximity"] = (
            pair["distance_cm"] <= PROXIMITY_THRESHOLD_CM
        )

        simultaneous_frames = len(pair)
        proximity_frames = int(pair["in_proximity"].sum())

        proximity_time_s = proximity_frames / FPS

        proximity_percent = (
            100 * proximity_frames / simultaneous_frames
            if simultaneous_frames > 0
            else np.nan
        )

        mean_pair_distance_cm = (
            pair["distance_cm"].mean()
            if simultaneous_frames > 0
            else np.nan
        )

        result = {
            "recording": recording,
            "mouse_a": mouse_a,
            "mouse_b": mouse_b,
            "simultaneously_resolved_frames": simultaneous_frames,
            "proximity_frames": proximity_frames,
            "proximity_time_s": proximity_time_s,
            "proximity_percent": proximity_percent,
            "mean_pair_distance_cm": mean_pair_distance_cm,
        }

        pair_results.append(result)
        all_pair_results.append(result)

    pair_out = pd.DataFrame(pair_results)

    pair_out.to_csv(
        OUTPUT / f"{recording}_pairwise_proximity.csv",
        index=False
    )

    print("\nPairwise proximity:")
    if not pair_out.empty:
        print(
            pair_out[
                [
                    "mouse_a",
                    "mouse_b",
                    "simultaneously_resolved_frames",
                    "proximity_time_s",
                    "proximity_percent",
                    "mean_pair_distance_cm",
                ]
            ].round(2)
        )


pd.DataFrame(all_animal_results).to_csv(
    OUTPUT / "all_recordings_animal_behaviour.csv",
    index=False
)

pd.DataFrame(all_pair_results).to_csv(
    OUTPUT / "all_recordings_pairwise_proximity.csv",
    index=False
)

print("\nComplete.")
print(f"Results saved to: {OUTPUT}")
