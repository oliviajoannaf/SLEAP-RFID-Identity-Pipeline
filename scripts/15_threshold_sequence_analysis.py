from pathlib import Path
import subprocess
import pandas as pd

BASE = Path("/lustre/biologie/franciso/Humboldt_Thesis")

SCRIPT12 = BASE / "scripts/12_propagate_rfid_identity.py"

RECORDING = "407405spareat1313"

ANCHORS = BASE / "outputs/development_v1/identity/407405spareat1313_rfid_reliable_anchors.csv"

DETECTIONS = BASE / "outputs/development_v1/tables/407405spareat1313_sleap_coordinates.csv"

OUTPUT = BASE / "outputs/development_v1/identity"

thresholds = [
    0.65,
    0.64,
    0.63,
    0.62,
    0.61,
    0.60,
]

summary_rows=[]

def classify(sequence, dominant):

    if pd.isna(sequence):
        return "no_reliable_anchors"

    seq=[s.strip() for s in str(sequence).split(",")]

    dominant=str(int(float(dominant)))

    conflicts=sum(x!=dominant for x in seq)

    transitions=sum(seq[i]!=seq[i-1] for i in range(1,len(seq)))

    if len(set(seq))==1:
        return "perfect_consistency"

    if conflicts==1:
        idx=[i for i,x in enumerate(seq) if x!=dominant][0]
        if idx==0 or idx==len(seq)-1:
            return "single_boundary_outlier"
        return "single_internal_outlier"

    if transitions==1:
        return "sustained_identity_transition"

    return "alternating_or_fragmented"


for t in thresholds:

    print(f"\nRunning {t:.2f}")

    subprocess.run([
        "python",
        str(SCRIPT12),
        "--recording",RECORDING,
        "--anchors",str(ANCHORS),
        "--detections",str(DETECTIONS),
        "--output-dir",str(OUTPUT),
        "--min-agreement",str(t)
    ],check=True)

    df=pd.read_csv(
        OUTPUT/f"{RECORDING}_identity_propagation_track_summary.csv"
    )

    df["pattern"] = [
        classify(s, d)
        for s, d in zip(
            df.anchor_sequence,
            df.dominant_mouse_id,
        )
    ]

    accepted = df[
        df["identity_is_known"] == True
    ].copy()

    summary_rows.append({

        "threshold": int(t * 100),

        "assigned_tracks":
        len(accepted),

        "perfect":
        (
            accepted.pattern
            == "perfect_consistency"
        ).sum(),

        "boundary_outlier":
        (
            accepted.pattern
            == "single_boundary_outlier"
        ).sum(),

        "transition":
        (
            accepted.pattern
            == "sustained_identity_transition"
        ).sum(),

        "no_anchor":
        (
            accepted.pattern
            == "no_reliable_anchors"
        ).sum()

    })

summary=pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT / "threshold_anchor_sequence_summary_fine_60_65.csv",
    index=False,
)

print(summary)
