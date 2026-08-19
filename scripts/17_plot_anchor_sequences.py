from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


base = Path(
    "/lustre/biologie/franciso/Humboldt_Thesis"
)

input_path = (
    base
    / "outputs/development_v1/identity"
    / "407405spareat1313_anchor_sequence_classification_75pct.csv"
)

figure_dir = (
    base
    / "outputs/development_v1/figures"
    / "threshold_optimisation"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = figure_dir / "example_chronological_anchor_sequences.png"


def parse_sequence(value):
    """Convert a comma-separated sequence to a list of strings."""
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def parse_frames(value):
    """Convert comma-separated frame indices to integers."""
    if pd.isna(value):
        return []

    frames = []

    for item in str(value).split(","):
        item = item.strip()

        if not item:
            continue

        frames.append(
            int(float(item))
        )

    return frames


df = pd.read_csv(input_path)

selected_tracks = [
    "track_12",
    "track_28",
    "track_38",
    "track_41",
]

selected = (
    df.loc[
        df["track_id"].isin(selected_tracks)
    ]
    .set_index("track_id")
    .reindex(selected_tracks)
    .reset_index()
)


fig, axes = plt.subplots(
    nrows=len(selected),
    ncols=1,
    figsize=(9, 8),
    sharex=False,
)

for ax, (_, row) in zip(
    axes,
    selected.iterrows(),
):
    sequence = parse_sequence(
        row["anchor_sequence"]
    )

    frames = parse_frames(
        row["anchor_frame_sequence"]
    )

    if len(sequence) != len(frames):
        raise ValueError(
            f"Sequence/frame mismatch for {row['track_id']}: "
            f"{len(sequence)} identities versus {len(frames)} frames."
        )

    mouse_ids = sorted(
        set(sequence)
    )

    y_lookup = {
        mouse_id: index
        for index, mouse_id in enumerate(mouse_ids)
    }

    y_values = [
        y_lookup[mouse_id]
        for mouse_id in sequence
    ]

    ax.plot(
        frames,
        y_values,
        marker="o",
        linestyle="-",
    )

    ax.set_yticks(
        list(y_lookup.values())
    )

    ax.set_yticklabels(
        list(y_lookup.keys())
    )

    ax.set_ylabel("Mouse ID")

    agreement = row["anchor_agreement"]
    pattern = row["anchor_pattern"].replace("_", " ")

    ax.set_title(
        f"{row['track_id']}: {pattern} "
        f"(agreement = {agreement:.2f})",
        loc="left",
    )

    ax.grid(axis="x", alpha=0.3)


axes[-1].set_xlabel("Video frame")

fig.suptitle(
    "Chronological RFID anchor evidence for conflicting SLEAP tracks",
    y=1.01,
)

fig.tight_layout()

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", output_path)
