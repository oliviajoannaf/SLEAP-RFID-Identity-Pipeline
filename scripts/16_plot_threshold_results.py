from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


base = Path(
    "/lustre/biologie/franciso/Humboldt_Thesis"
)

input_path = (
    base
    / "outputs/development_v1/identity"
    / "threshold_optimisation_final_table.csv"
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

df = pd.read_csv(input_path).sort_values("threshold")


# ---------------------------------------------------------
# Figure 1: Agreement threshold versus accepted tracks
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    df["threshold"],
    df["assigned_tracks"],
    marker="o",
)

ax.set_xlabel("Minimum RFID agreement threshold (%)")
ax.set_ylabel("Accepted SLEAP tracks")
ax.set_title("Effect of RFID agreement threshold on identity recovery")

ax.set_xticks(df["threshold"])
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

output_1 = figure_dir / "threshold_vs_accepted_tracks.png"

fig.savefig(
    output_1,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------
# Figure 2: Agreement threshold versus labelled detections
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    df["threshold"],
    df["labelled_detections_pct"],
    marker="o",
)

ax.set_xlabel("Minimum RFID agreement threshold (%)")
ax.set_ylabel("Identity-labelled detections (%)")
ax.set_title("Effect of RFID agreement threshold on detection coverage")

ax.set_xticks(df["threshold"])
ax.set_ylim(
    max(0, df["labelled_detections_pct"].min() - 5),
    min(100, df["labelled_detections_pct"].max() + 5),
)

ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

output_2 = figure_dir / "threshold_vs_labelled_detections.png"

fig.savefig(
    output_2,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print("Saved:")
print(" ", output_1)
print(" ", output_2)
