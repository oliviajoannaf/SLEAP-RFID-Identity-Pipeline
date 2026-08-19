#!/bin/bash

set -e

RECORDING=$1
TIMESTAMP_BASE=$2

echo "======================================"
echo "Processing ${RECORDING}"
echo "======================================"

TABLE_DIR="outputs/experimental_v2/tables"
RFID_DIR="outputs/experimental_v2/rfid"
IDENTITY_DIR="outputs/experimental_v2/identity"

python scripts/05_extract_sleap_coordinates.py \
outputs/experimental_v2/${RECORDING}.predictions.slp \
${TABLE_DIR}/${RECORDING}_sleap.csv

python scripts/06_align_timestamps.py \
${TABLE_DIR}/${RECORDING}_sleap.csv \
videos/TESTsleapvideos/${TIMESTAMP_BASE}_1.csv \
${TABLE_DIR}/${RECORDING}_timestamped.csv

python scripts/09_extract_video_rfid.py \
${TABLE_DIR}/${RECORDING}_timestamped.csv \
${RFID_DIR}/26.07.14.RFID_rfid_events_mapped.csv \
${RFID_DIR}/${RECORDING}_video_rfid.csv \
${RFID_DIR}/${RECORDING}_video_rfid_summary.txt

python scripts/11_assign_rfid_anchors.py \
--video-id ${RECORDING} \
--sleap ${TABLE_DIR}/${RECORDING}_timestamped.csv \
--rfid ${RFID_DIR}/${RECORDING}_video_rfid.csv \
--reader-cm config/rfid_reader_positions_cm_basler_180.csv \
--calibration config/arena_calibration_407405spareat1313.csv \
--output-dir ${IDENTITY_DIR}

python scripts/11_filter_rfid_anchors.py \
--input ${IDENTITY_DIR}/${RECORDING}_rfid_anchor_assignments.csv \
--output-dir ${IDENTITY_DIR} \
--recording-id ${RECORDING}

python scripts/12_propagate_rfid_identity.py \
--recording ${RECORDING} \
--anchors ${IDENTITY_DIR}/${RECORDING}_rfid_reliable_anchors.csv \
--detections ${TABLE_DIR}/${RECORDING}_timestamped.csv \
--output-dir ${IDENTITY_DIR}

echo ""
echo "Finished ${RECORDING}"
