#!/usr/bin/env bash

set -euo pipefail

IDENTITY_DIR="outputs/experimental_v2/identity"
TABLE_DIR="outputs/experimental_v2/tables"
SWEEP_DIR="outputs/experimental_v2/threshold_sweep"
PROPAGATION_SCRIPT="scripts/12_propagate_rfid_identity.py"

thresholds=(50 55 60 65 70 75 80 85 90 95 100)

mkdir -p "$SWEEP_DIR"

shopt -s nullglob
anchor_files=("$IDENTITY_DIR"/*_rfid_reliable_anchors.csv)

if [ "${#anchor_files[@]}" -eq 0 ]; then
    echo "ERROR: No reliable-anchor files found in $IDENTITY_DIR"
    exit 1
fi

echo "Found ${#anchor_files[@]} experimental recordings."
echo

for anchor_file in "${anchor_files[@]}"; do
    filename=$(basename "$anchor_file")
    recording=${filename%_rfid_reliable_anchors.csv}
    detections="$TABLE_DIR/${recording}_sleap.csv"

    if [ ! -f "$detections" ]; then
        echo "ERROR: Missing SLEAP table for $recording"
        echo "Expected: $detections"
        exit 1
    fi

    echo "Recording: $recording"
    echo "--------------------------------------------"

    for threshold_pct in "${thresholds[@]}"; do
        threshold_decimal=$(awk \
            -v value="$threshold_pct" \
            'BEGIN {printf "%.2f", value / 100}'
        )

        threshold_name=$(printf "threshold_%03d" "$threshold_pct")
        output_dir="$SWEEP_DIR/$threshold_name"
        log_dir="$output_dir/logs"

        mkdir -p "$output_dir" "$log_dir"

        echo "Running ${threshold_pct}% agreement..."

        python "$PROPAGATION_SCRIPT" \
            --recording "$recording" \
            --anchors "$anchor_file" \
            --detections "$detections" \
            --output-dir "$output_dir" \
            --min-agreement "$threshold_decimal" \
            > "$log_dir/${recording}.log" 2>&1

        echo "  Completed."
    done

    echo
done

echo "Threshold sweep completed successfully."
echo "Outputs saved under: $SWEEP_DIR"
