from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CHANGE THIS PATH
# ============================================================

INPUT_FILE = Path(
    "outputs/experimental_v2/PUT_YOUR_RFID_ANCHOR_FILE_HERE.csv"
)

OUTPUT_DIR = Path(
    "outputs/experimental_v2/rfid_identity_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Use:\nfind outputs -type f -name '*.csv' | sort\n"
        "to locate the RFID-anchor assignment CSV."
    )

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())


# ============================================================
# AUTOMATIC COLUMN DETECTION
# ============================================================

def find_column(candidates, required=True):
    lower_map = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            "\nCould not find one of these required columns:\n"
            f"{candidates}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    return None


recording_col = find_column(
    [
        "recording",
        "video",
        "video_name",
        "recording_name",
        "file",
        "filename",
    ],
    required=False,
)

track_col = find_column(
    [
        "track_id",
        "track",
        "track_name",
        "sleap_track",
        "sleap_track_id",
        "instance_track",
        "predicted_track",
    ]
)

rfid_mouse_col = find_column(
    [
        "rfid_mouse_id",
        "rfid_identity",
        "rfid_id",
        "anchor_mouse_id",
        "anchor_identity",
        "animal_id",
        "mouse_id",
        "mouse",
    ]
)

frame_col = find_column(
    [
        "frame_idx",
        "frame",
        "frame_index",
        "video_frame",
    ],
    required=False,
)


# If no recording column exists, treat the entire file as one recording.
if recording_col is None:
    df["_recording"] = INPUT_FILE.stem
    recording_col = "_recording"


# ============================================================
# CLEAN DATA
# ============================================================

anchors = df[
    [recording_col, track_col, rfid_mouse_col]
    + ([frame_col] if frame_col is not None else [])
].copy()

anchors = anchors.dropna(
    subset=[recording_col, track_col, rfid_mouse_col]
)

anchors[recording_col] = anchors[recording_col].astype(str)
anchors[track_col] = anchors[track_col].astype(str)
anchors[rfid_mouse_col] = anchors[rfid_mouse_col].astype(str)

print(f"\nRFID anchor observations: {len(anchors):,}")
print(
    "Unique recordings:",
    anchors[recording_col].nunique()
)
print(
    "Unique SLEAP tracks:",
    anchors[[recording_col, track_col]]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# ANALYSIS 1:
# RFID IDENTITIES ASSOCIATED WITH EACH SLEAP TRACK
# ============================================================

track_summary = (
    anchors
    .groupby([recording_col, track_col])
    .agg(
        anchor_count=(rfid_mouse_col, "size"),
        unique_rfid_identities=(rfid_mouse_col, "nunique"),
        rfid_identities=(
            rfid_mouse_col,
            lambda x: ", ".join(sorted(set(x)))
        ),
        majority_identity=(
            rfid_mouse_col,
            lambda x: x.value_counts().index[0]
        ),
        majority_identity_count=(
            rfid_mouse_col,
            lambda x: int(x.value_counts().iloc[0])
        ),
    )
    .reset_index()
)

track_summary["identity_conflict"] = (
    track_summary["unique_rfid_identities"] > 1
)

track_summary["majority_identity_proportion"] = (
    100
    * track_summary["majority_identity_count"]
    / track_summary["anchor_count"]
)

track_summary.to_csv(
    OUTPUT_DIR / "track_level_rfid_identity_consistency.csv",
    index=False,
)


# ============================================================
# SUMMARY BY RECORDING
# ============================================================

recording_summary = (
    track_summary
    .groupby(recording_col)
    .agg(
        anchored_sleap_tracks=(track_col, "nunique"),
        consistent_tracks=(
            "identity_conflict",
            lambda x: int((~x).sum())
        ),
        conflicted_tracks=(
            "identity_conflict",
            lambda x: int(x.sum())
        ),
        tracks_with_multiple_anchors=(
            "anchor_count",
            lambda x: int((x >= 2).sum())
        ),
        median_anchors_per_track=("anchor_count", "median"),
        median_majority_agreement_percent=(
            "majority_identity_proportion",
            "median"
        ),
    )
    .reset_index()
)

recording_summary["conflicted_tracks_percent"] = (
    100
    * recording_summary["conflicted_tracks"]
    / recording_summary["anchored_sleap_tracks"]
)

recording_summary.to_csv(
    OUTPUT_DIR / "recording_level_track_conflicts.csv",
    index=False,
)


# ============================================================
# ANALYSIS 2:
# LEAVE-ONE-ANCHOR-OUT VALIDATION
#
# For each RFID anchor:
# - remove that observation;
# - use the remaining RFID observations on the same SLEAP track;
# - predict the omitted identity using the majority identity.
#
# Tracks with only one RFID anchor cannot be evaluated.
# Ties are marked unresolved rather than guessed.
# ============================================================

validation_rows = []

group_columns = [recording_col, track_col]

for group_key, group in anchors.groupby(group_columns):

    group = group.reset_index(drop=True)

    if len(group) < 2:
        continue

    for held_out_index in range(len(group)):

        held_out = group.iloc[held_out_index]
        training = group.drop(index=held_out_index)

        counts = training[rfid_mouse_col].value_counts()

        if len(counts) == 0:
            continue

        top_count = counts.iloc[0]
        top_identities = counts[counts == top_count].index.tolist()

        # Do not guess when two identities are equally supported.
        if len(top_identities) != 1:
            predicted_identity = np.nan
            prediction_status = "tie"
            correct = np.nan
        else:
            predicted_identity = str(top_identities[0])
            prediction_status = "predicted"
            correct = (
                predicted_identity
                == str(held_out[rfid_mouse_col])
            )

        row = {
            recording_col: held_out[recording_col],
            track_col: held_out[track_col],
            "true_rfid_identity":
                str(held_out[rfid_mouse_col]),
            "predicted_rfid_identity":
                predicted_identity,
            "prediction_status":
                prediction_status,
            "correct":
                correct,
            "remaining_anchor_count":
                len(training),
        }

        if frame_col is not None:
            row[frame_col] = held_out[frame_col]

        validation_rows.append(row)


validation = pd.DataFrame(validation_rows)

if validation.empty:
    print(
        "\nNo tracks had at least two RFID anchors, "
        "so leave-one-anchor-out validation could not be performed."
    )

else:
    validation.to_csv(
        OUTPUT_DIR / "leave_one_anchor_out_predictions.csv",
        index=False,
    )

    evaluable = validation[
        validation["prediction_status"] == "predicted"
    ].copy()

    evaluable["correct"] = evaluable["correct"].astype(bool)

    validation_summary = (
        evaluable
        .groupby(recording_col)
        .agg(
            evaluated_anchors=("correct", "size"),
            correct_predictions=("correct", "sum"),
        )
        .reset_index()
    )

    validation_summary["incorrect_predictions"] = (
        validation_summary["evaluated_anchors"]
        - validation_summary["correct_predictions"]
    )

    validation_summary["identity_accuracy_percent"] = (
        100
        * validation_summary["correct_predictions"]
        / validation_summary["evaluated_anchors"]
    )

    overall = pd.DataFrame([{
        recording_col: "OVERALL",
        "evaluated_anchors": len(evaluable),
        "correct_predictions": int(evaluable["correct"].sum()),
        "incorrect_predictions": int((~evaluable["correct"]).sum()),
        "identity_accuracy_percent":
            100 * evaluable["correct"].mean()
            if len(evaluable) > 0 else np.nan,
    }])

    validation_summary = pd.concat(
        [validation_summary, overall],
        ignore_index=True,
    )

    validation_summary.to_csv(
        OUTPUT_DIR / "leave_one_anchor_out_accuracy.csv",
        index=False,
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion = pd.crosstab(
        evaluable["true_rfid_identity"],
        evaluable["predicted_rfid_identity"],
        rownames=["True RFID identity"],
        colnames=["Predicted identity"],
        dropna=False,
    )

    confusion.to_csv(
        OUTPUT_DIR / "leave_one_anchor_out_confusion_matrix.csv"
    )


    # ========================================================
    # CONFUSION MATRIX FIGURE
    # ========================================================

    if not confusion.empty:

        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(confusion.values)

        ax.set_xticks(range(len(confusion.columns)))
        ax.set_xticklabels(
            confusion.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(range(len(confusion.index)))
        ax.set_yticklabels(confusion.index)

        ax.set_xlabel("Predicted RFID identity")
        ax.set_ylabel("True RFID identity")
        ax.set_title(
            "Leave-one-anchor-out identity validation"
        )

        for row_index in range(confusion.shape[0]):
            for column_index in range(confusion.shape[1]):
                ax.text(
                    column_index,
                    row_index,
                    int(
                        confusion.iloc[
                            row_index,
                            column_index
                        ]
                    ),
                    ha="center",
                    va="center",
                )

        fig.colorbar(image, ax=ax, label="RFID anchors")
        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR
            / "Figure_leave_one_anchor_out_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# FIGURE: CONFLICTED TRACKS
# ============================================================

plot_data = recording_summary.copy()

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    plot_data[recording_col],
    plot_data["conflicted_tracks_percent"],
)

ax.set_ylabel("RFID-conflicted SLEAP tracks (%)")
ax.set_xlabel("Recording")
ax.set_title(
    "SLEAP tracks associated with multiple RFID identities"
)

ax.tick_params(
    axis="x",
    labelrotation=25,
)

for bar, value in zip(
    bars,
    plot_data["conflicted_tracks_percent"],
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{value:.1f}%",
        ha="center",
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_RFID_conflicted_SLEAP_tracks.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# PRINT KEY RESULTS
# ============================================================

print("\nTRACK-LEVEL RFID CONSISTENCY")
print(recording_summary.to_string(index=False))

if not validation.empty:
    print("\nLEAVE-ONE-ANCHOR-OUT VALIDATION")
    print(validation_summary.to_string(index=False))

print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

print("\nCreated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)
