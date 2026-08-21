from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROPAGATED_DIR = Path(
    "outputs/experimental_v2/final_identity_threshold_075"
)

BEHAVIOUR_DIR = Path(
    "outputs/experimental_v2/behaviour_complete"
)

OUTPUT = Path(
    "outputs/experimental_v2/thesis_tables_figures"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# RECORDING LABELS
# ============================================================

RECORDING_INFO = {
    "4074408TE1at1040test_v2": {
        "label": "2 mice: 407 + 408",
        "group_size": 2,
        "expected_mice": 2,
    },
    "405408at1020_v2": {
        "label": "2 mice: 405 + 408",
        "group_size": 2,
        "expected_mice": 2,
    },
    "406407408at1126_v2": {
        "label": "3 mice: 406 + 407 + 408",
        "group_size": 3,
        "expected_mice": 3,
    },
    "405046407408x4miceat1244_v2": {
        "label": "4 mice: 405 + 406 + 407 + 408",
        "group_size": 4,
        "expected_mice": 4,
    },
}


# ============================================================
# TABLE 4.1: SLEAP TRACK FRAGMENTATION
# ============================================================

# These values came from the SLEAP track-length analysis.
# One representative recording is included for each group size.

fragmentation = pd.DataFrame([
    {
        "Recording": "2-mouse recording",
        "Mice": 2,
        "Frames": 32255,
        "SLEAP tracks": 242,
        "Tracks >1000 frames": 10,
        "Tracks <30 frames": 70,
        "Median track length": 85,
        "Longest track": 2140,
    },
    {
        "Recording": "3-mouse recording",
        "Mice": 3,
        "Frames": 27631,
        "SLEAP tracks": 295,
        "Tracks >1000 frames": 15,
        "Tracks <30 frames": 75,
        "Median track length": 144,
        "Longest track": 1933,
    },
    {
        "Recording": "4-mouse recording",
        "Mice": 4,
        "Frames": 28584,
        "SLEAP tracks": 431,
        "Tracks >1000 frames": 17,
        "Tracks <30 frames": 116,
        "Median track length": 105,
        "Longest track": 2628,
    },
])

fragmentation["Tracks per biological animal"] = (
    fragmentation["SLEAP tracks"] / fragmentation["Mice"]
)

fragmentation["Short-track percentage"] = (
    100
    * fragmentation["Tracks <30 frames"]
    / fragmentation["SLEAP tracks"]
)

fragmentation.to_csv(
    OUTPUT / "Table_4_1_SLEAP_track_fragmentation.csv",
    index=False,
)


# ============================================================
# FIGURE: TOTAL SLEAP TRACKS
# ============================================================

plt.figure(figsize=(7, 5))

bars = plt.bar(
    fragmentation["Recording"],
    fragmentation["SLEAP tracks"],
)

plt.ylabel("Number of SLEAP track identities")
plt.xlabel("Recording group size")
plt.title("SLEAP track fragmentation across group sizes")

for bar, value in zip(bars, fragmentation["SLEAP tracks"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        str(value),
        ha="center",
    )

plt.ylim(0, fragmentation["SLEAP tracks"].max() * 1.18)
plt.tight_layout()

plt.savefig(
    OUTPUT / "Figure_SLEAP_total_tracks_by_group_size.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE: TRACKS PER BIOLOGICAL ANIMAL
# ============================================================

plt.figure(figsize=(7, 5))

bars = plt.bar(
    fragmentation["Recording"],
    fragmentation["Tracks per biological animal"],
)

plt.ylabel("SLEAP track identities per biological animal")
plt.xlabel("Recording group size")
plt.title("Normalised SLEAP identity fragmentation")

for bar, value in zip(
    bars,
    fragmentation["Tracks per biological animal"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{value:.1f}",
        ha="center",
    )

plt.ylim(
    0,
    fragmentation["Tracks per biological animal"].max() * 1.18,
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "Figure_SLEAP_tracks_per_animal.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# TABLE 4.2: RFID IDENTITY PROPAGATION
# ============================================================

identity_results = []

for file in sorted(PROPAGATED_DIR.glob("*_identity_propagated.csv")):

    recording = file.name.replace(
        "_identity_propagated.csv",
        ""
    )

    df = pd.read_csv(file)

    if recording not in RECORDING_INFO:
        print(f"Skipping unknown recording: {recording}")
        continue

    info = RECORDING_INFO[recording]

    known_mask = (
        df["identity_is_known"]
        .fillna(False)
        .astype(bool)
    )

    total_detections = len(df)
    resolved_detections = int(known_mask.sum())

    resolved_percentage = (
        100 * resolved_detections / total_detections
        if total_detections > 0
        else np.nan
    )

    final_ids = sorted(
        df.loc[known_mask, "mouse_id"]
        .dropna()
        .astype(int)
        .unique()
    )

    total_video_frames = int(df["frame_idx"].nunique())

    frames_with_resolved_identity = int(
        df.loc[known_mask, "frame_idx"].nunique()
    )

    frame_coverage_percentage = (
        100
        * frames_with_resolved_identity
        / total_video_frames
        if total_video_frames > 0
        else np.nan
    )

    identity_results.append({
        "Recording": info["label"],
        "Group size": info["group_size"],
        "Expected biological mice": info["expected_mice"],
        "Final biological identities": len(final_ids),
        "Mouse IDs": ", ".join(map(str, final_ids)),
        "Total detections": total_detections,
        "Identity-resolved detections": resolved_detections,
        "Resolved detections (%)": resolved_percentage,
        "Video frames represented": total_video_frames,
        "Frames with at least one resolved identity":
            frames_with_resolved_identity,
        "Frame coverage (%)": frame_coverage_percentage,
    })

identity = pd.DataFrame(identity_results)

identity = identity.sort_values(
    ["Group size", "Recording"]
).reset_index(drop=True)

identity.to_csv(
    OUTPUT / "Table_4_2_RFID_identity_propagation.csv",
    index=False,
)


# ============================================================
# FIGURE: IDENTITY-RESOLVED DETECTIONS
# ============================================================

plt.figure(figsize=(9, 5))

bars = plt.bar(
    identity["Recording"],
    identity["Resolved detections (%)"],
)

plt.ylabel("Identity-resolved detections (%)")
plt.xlabel("Recording")
plt.title("RFID-assisted identity resolution")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 100)

for bar, value in zip(
    bars,
    identity["Resolved detections (%)"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{value:.1f}%",
        ha="center",
    )

plt.tight_layout()

plt.savefig(
    OUTPUT / "Figure_RFID_identity_resolved_percentage.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# TABLE 4.3: INDIVIDUAL BEHAVIOURAL METRICS
# ============================================================

behaviour_file = (
    BEHAVIOUR_DIR
    / "all_recordings_animal_behaviour.csv"
)

behaviour = pd.read_csv(behaviour_file)

behaviour["Recording label"] = behaviour["recording"].map(
    lambda x: RECORDING_INFO.get(
        x,
        {"label": x}
    )["label"]
)

behaviour_table = behaviour[[
    "Recording label",
    "mouse_id",
    "identity_resolved_frames",
    "identity_resolved_time_s",
    "distance_cm",
    "mean_speed_cm_s",
    "centre_time_s",
    "centre_occupancy_percent",
]].copy()

behaviour_table.columns = [
    "Recording",
    "Mouse",
    "Identity-resolved frames",
    "Identity-resolved time (s)",
    "Distance travelled (cm)",
    "Mean speed (cm/s)",
    "Centre time (s)",
    "Centre occupancy (%)",
]

behaviour_table["Mouse"] = (
    behaviour_table["Mouse"].astype(int)
)

for column in [
    "Identity-resolved time (s)",
    "Distance travelled (cm)",
    "Mean speed (cm/s)",
    "Centre time (s)",
    "Centre occupancy (%)",
]:
    behaviour_table[column] = (
        behaviour_table[column].round(2)
    )

behaviour_table.to_csv(
    OUTPUT / "Table_4_3_individual_behaviour.csv",
    index=False,
)


# ============================================================
# FIGURE: DISTANCE TRAVELLED
# ============================================================

behaviour_plot = behaviour_table.copy()

behaviour_plot["Animal"] = (
    behaviour_plot["Recording"]
    + " — Mouse "
    + behaviour_plot["Mouse"].astype(str)
)

behaviour_plot = behaviour_plot.sort_values(
    "Distance travelled (cm)"
)

plt.figure(figsize=(10, 7))

plt.barh(
    behaviour_plot["Animal"],
    behaviour_plot["Distance travelled (cm)"],
)

plt.xlabel("Distance travelled (cm)")
plt.ylabel("")
plt.title("RFID-resolved distance travelled by individual mice")
plt.tight_layout()

plt.savefig(
    OUTPUT / "Figure_distance_travelled_by_mouse.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE: CENTRE OCCUPANCY
# ============================================================

behaviour_plot = behaviour_table.copy()

behaviour_plot["Animal"] = (
    behaviour_plot["Recording"]
    + " — Mouse "
    + behaviour_plot["Mouse"].astype(str)
)

behaviour_plot = behaviour_plot.sort_values(
    "Centre occupancy (%)"
)

plt.figure(figsize=(10, 7))

plt.barh(
    behaviour_plot["Animal"],
    behaviour_plot["Centre occupancy (%)"],
)

plt.xlabel("Centre occupancy (%)")
plt.ylabel("")
plt.title("RFID-resolved centre occupancy by individual mice")
plt.tight_layout()

plt.savefig(
    OUTPUT / "Figure_centre_occupancy_by_mouse.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# TABLE 4.4: PAIRWISE PROXIMITY
# ============================================================

pair_file = (
    BEHAVIOUR_DIR
    / "all_recordings_pairwise_proximity.csv"
)

pairs = pd.read_csv(pair_file)

pairs["Recording label"] = pairs["recording"].map(
    lambda x: RECORDING_INFO.get(
        x,
        {"label": x}
    )["label"]
)

pairs["Pair"] = (
    pairs["mouse_a"].astype(int).astype(str)
    + "–"
    + pairs["mouse_b"].astype(int).astype(str)
)

pair_table = pairs[[
    "Recording label",
    "Pair",
    "simultaneously_resolved_frames",
    "proximity_time_s",
    "proximity_percent",
    "mean_pair_distance_cm",
]].copy()

pair_table.columns = [
    "Recording",
    "Mouse pair",
    "Simultaneously resolved frames",
    "Proximity time (s)",
    "Proximity (%)",
    "Mean pairwise distance (cm)",
]

for column in [
    "Proximity time (s)",
    "Proximity (%)",
    "Mean pairwise distance (cm)",
]:
    pair_table[column] = pair_table[column].round(2)

pair_table.to_csv(
    OUTPUT / "Table_4_4_pairwise_proximity.csv",
    index=False,
)


# ============================================================
# FIGURE: PAIRWISE PROXIMITY
# ============================================================

pair_plot = pair_table.copy()

pair_plot["Recording pair"] = (
    pair_plot["Recording"]
    + " — "
    + pair_plot["Mouse pair"]
)

pair_plot = pair_plot.sort_values(
    "Proximity (%)"
)

plt.figure(figsize=(10, 7))

plt.barh(
    pair_plot["Recording pair"],
    pair_plot["Proximity (%)"],
)

plt.xlabel("Frames within 5 cm (%)")
plt.ylabel("")
plt.title("RFID-resolved pairwise proximity")
plt.tight_layout()

plt.savefig(
    OUTPUT / "Figure_pairwise_proximity.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# PIPELINE SUMMARY TABLE
# ============================================================

pipeline_summary = pd.DataFrame([
    {
        "Pipeline stage": "SLEAP pose tracking",
        "Output": "Animal detections and pose coordinates",
        "Principal limitation":
            "Hundreds of fragmented anonymous track identities",
    },
    {
        "Pipeline stage": "RFID anchor assignment",
        "Output": "Sparse observations of known biological identity",
        "Principal limitation":
            "Identity available only near RFID antenna detections",
    },
    {
        "Pipeline stage": "Identity propagation",
        "Output": "Track fragments assigned to biological mice",
        "Principal limitation":
            "Unresolved intervals retained where evidence was insufficient",
    },
    {
        "Pipeline stage": "Behavioural analysis",
        "Output":
            "Distance, speed, centre occupancy and pairwise proximity",
        "Principal limitation":
            "Metrics calculated only from identity-resolved observations",
    },
])

pipeline_summary.to_csv(
    OUTPUT / "Pipeline_summary_table.csv",
    index=False,
)


# ============================================================
# FINISHED
# ============================================================

print("\nCreated thesis outputs:")
for file in sorted(OUTPUT.iterdir()):
    print(file)

print(f"\nAll outputs saved to:\n{OUTPUT}")
