# SLEAP–RFID Identity Reconstruction Pipeline

This repository contains the computational workflow developed for RFID-assisted reconstruction of biological identity from multi-animal SLEAP pose trajectories within the AI Open Arena at Humboldt-Universität zu Berlin.

## Overview

SLEAP provides high-resolution markerless pose estimates for multiple freely interacting animals, but temporary visual track identities may become fragmented during interactions, overlap and occlusion.

This workflow incorporates radio-frequency identification (RFID) downstream of SLEAP as an independent source of biological identity. Sparse RFID observations are temporally synchronised with video, spatially associated with SLEAP detections and used as identity anchors. Consistent RFID evidence is subsequently propagated across supported SLEAP track fragments, while insufficiently supported or conflicting fragments remain unresolved.

## Computational workflow

1. SLEAP model configuration and training
2. SLEAP inference on experimental recordings
3. Extraction of frame-wise pose coordinates
4. Basler timestamp alignment
5. RFID preprocessing and biological identity mapping
6. Video-specific RFID temporal alignment
7. RFID reader spatial calibration
8. RFID–SLEAP anchor assignment and quality control
9. RFID-assisted biological identity propagation
10. Identity threshold validation and quality control
11. Identity-resolved behavioural analysis
12. Generation of thesis summary tables and figures

## SLEAP configuration

A top-down SLEAP architecture was used with separate centroid-detection and centred-instance models.

Five anatomical landmarks were tracked:

- nose
- left ear
- right ear
- torso centre
- tail base

Final SLEAP training configurations are provided in `training_configs/`.

## RFID-assisted identity reconstruction

RFID detections provide intermittent observations of biological identity independently of visual appearance. Following temporal and spatial alignment, reliable RFID observations associated with a SLEAP track are aggregated as identity evidence.

Biological identity is propagated across a track fragment only where RFID evidence satisfies the validated agreement criterion. The final workflow uses a 75% minimum RFID agreement threshold. Tracks without sufficient identity evidence remain unresolved rather than receiving forced biological assignments.

## Behavioural outputs

Identity-resolved trajectories can be used to calculate individual- and pair-specific measures including:

- distance travelled
- centre occupancy
- pairwise proximity

## Repository structure

`/scripts`  
Final processing, identity reconstruction, validation and behavioural-analysis scripts.

`/training_configs`  
Final SLEAP centroid and centred-instance training configurations.

`/configs`  
Arena and RFID reader spatial-calibration files.

`/documentation`  
Description of the computational workflow and execution stages.

## Data availability

Raw experimental video, RFID logs, trained model files and animal-specific RFID tag mappings are not included in this repository.

## Thesis context

This workflow was developed as part of an MSc MorphoPHEN thesis within the AI Open Arena project at the Winter Laboratory, Humboldt-Universität zu Berlin.

## Reuse and reproducibility

Several scripts retain project-specific recording identifiers, relative output locations and Humboldt HPC paths used for the analyses reported in the accompanying thesis. These values document the executed research workflow and should be replaced with local dataset paths when adapting the pipeline to new recordings.

The repository does not contain the original animal-specific RFID transponder mapping. An example mapping schema is provided at `configs/rfid_tag_mapping_example.csv`.


## Running the identity pipeline

The repository includes `scripts/run_identity_pipeline.sh` to execute the principal identity-reconstruction stages for an experimental recording after SLEAP inference and RFID preprocessing.

The runner assumes:
- a SLEAP prediction file has already been generated;
- RFID timestamps have been converted and transponder identifiers mapped to biological mouse IDs;
- project-specific recording names and input paths have been configured locally.

Threshold sensitivity analyses can subsequently be executed using `scripts/run_identity_threshold_sweep.sh`.

Detailed stage-by-stage inputs, outputs and script functions are described in `documentation/workflow.md`.
