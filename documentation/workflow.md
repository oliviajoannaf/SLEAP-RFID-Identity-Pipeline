# Computational workflow

## Stage 1 — SLEAP model configuration and training

Scripts:
- `02_generate_training_configs.sh`
- `03_train_topdown_models.sh`

Purpose:
Generate and train the final top-down SLEAP centroid-detection and centred-instance models.

Final model configuration files are provided in `training_configs/`.

---

## Stage 2 — SLEAP inference

Script:
- `04_run_inference.sh`

Purpose:
Apply the trained top-down SLEAP models to unseen experimental recordings.

Primary output:
- SLEAP prediction file (`.slp`)

---

## Stage 3 — Coordinate extraction

Script:
- `05_extract_sleap_coordinates.py`

Purpose:
Convert native SLEAP predictions into a frame-wise tabular coordinate format.

Retained information includes:
- frame index
- temporary SLEAP track identifier
- centroid coordinates
- anatomical landmark coordinates
- confidence scores
- landmark visibility information

---

## Stage 4 — Basler timestamp alignment

Script:
- `06_align_timestamps.py`

Purpose:
Associate every SLEAP detection with the absolute acquisition timestamp of its corresponding Basler video frame.

---

## Stage 5 — RFID preprocessing and identity mapping

Scripts:
- `07_convert_rfid_timestamps.py`
- `08_map_rfid_tags.py`

Purpose:
Separate RFID detection events from acquisition metadata, standardise RFID timestamps and map RFID transponder identifiers to biological mouse identities.

Animal-specific RFID mapping files are not distributed with this repository.

---

## Stage 6 — Video-specific RFID alignment

Script:
- `09_extract_video_rfid.py`

Purpose:
Extract RFID events overlapping each Basler recording and associate each RFID event with the nearest video frame.

---

## Stage 7 — RFID spatial calibration

Scripts:
- `10_overlay_rfid_readers.py`
- `11_plot_corrected_rfid_overlay.py`

Purpose:
Represent the physical RFID reader arrangement within the Basler image coordinate system and verify spatial registration between RFID reader positions and video-derived animal locations.

Supporting calibration files are provided in `configs/`.

---

## Stage 8 — RFID anchor assignment and quality control

Scripts:
- `11_assign_rfid_anchors.py`
- `11_filter_rfid_anchors.py`
- `11_validate_rfid_anchors.py`

Purpose:
Associate temporally aligned RFID detections with spatially compatible SLEAP torso-centre detections.

Provisional RFID anchors are filtered to reject spatially distant or ambiguous assignments before biological identity propagation.

---

## Stage 9 — Biological identity propagation

Script:
- `12_propagate_rfid_identity.py`

Purpose:
Aggregate reliable RFID identity evidence within each temporary SLEAP track fragment.

The dominant RFID-supported biological identity is propagated across a track only when the required minimum agreement criterion is satisfied.

Tracks without sufficient or consistent RFID evidence remain unresolved.

---

## Stage 10 — Identity validation and agreement-threshold analysis

Scripts:
- `13_validate_propagated_identity.py`
- `13_analyse_identity_threshold_sweep.py`
- `14_validate_identity_threshold_frames.py`
- `15_threshold_sequence_analysis.py`
- `16_plot_threshold_results.py`
- `17_plot_anchor_sequences.py`
- `20_validate_rfid_track_identity.py`

Purpose:
Evaluate identity reconstruction using:
- track-level identity agreement
- detection-level identity coverage
- frame-level identity coverage
- unresolved tracks
- conflicting biological identity evidence
- chronological RFID identity transitions

The final thesis workflow used a fixed 75% RFID agreement threshold selected within a stable operating region identified during development and independently evaluated on a three-mouse recording.

---

## Stage 11 — Identity-resolved behavioural analysis

Script:
- `18_complete_behaviour_analysis.py`

Purpose:
Derive individual- and pair-specific behavioural outputs from identity-resolved SLEAP trajectories.

Final behavioural measures include:
- distance travelled
- centre occupancy
- pairwise proximity

---

## Processing principle

RFID is used as a downstream biological identity layer rather than as a replacement for SLEAP pose estimation.

SLEAP supplies dense spatial and anatomical pose information, while RFID supplies sparse observations of biological identity independently of visual appearance.

Uncertain trajectory fragments are retained as unresolved rather than receiving forced identity assignments.
