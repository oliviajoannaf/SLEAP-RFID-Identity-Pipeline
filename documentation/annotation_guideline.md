# SLEAP Annotation Guideline

**Version 1.0**

This guideline defines the anatomical landmark conventions used to generate the manually annotated training data for the SLEAP multi-animal pose-estimation model. The same definitions were applied throughout annotation to maintain consistency across animals, postures, interactions and recording conditions.

## General principles

### Rule 1 — Maintain anatomical consistency

Place each node at the same defined anatomical location across all labelled frames, regardless of changes in posture, orientation or appearance.

### Rule 2 — Use the underlying anatomy as the reference

Changes in posture, fur and body orientation can alter the apparent shape of the animal. Landmark placement should therefore reflect the estimated underlying anatomical location rather than superficial changes in appearance.

### Rule 3 — Account for the overhead viewing perspective

Recordings were acquired from a fixed overhead camera. Nodes should correspond to the defined anatomical landmark rather than the apparent centre of shadows, reflections or other image artefacts.

### Rule 4 — Prioritise consistent anatomical definitions

When the precise position of a landmark is uncertain within a small spatial region, apply the same anatomical interpretation consistently across labelled frames rather than changing the landmark definition according to visual appearance.

---

## Landmark definitions

### 1. Nose

The nose node was defined as the most anterior point of the snout.

The node should not be positioned on the whiskers, surrounding fur or shadows. Where minor motion blur obscures the precise boundary of the nose, estimate the centre of the snout tip.

### 2. Left ear

The left-ear node was defined as the centre of the animal's anatomical left ear.

Left and right refer to the animal's anatomical perspective rather than the viewer's perspective. If ear orientation changes, retain the centre of the ear as the anatomical reference rather than moving the node towards its visible edge.

### 3. Right ear

The right-ear node was defined as the centre of the animal's anatomical right ear.

As for the left ear, right refers to the animal's anatomical perspective rather than the viewer's perspective.

### 4. Torso centre

The torso-centre node was defined as the approximate geometric centre of the trunk along the longitudinal body axis between the shoulder region and tail base.

The node should remain centred on the trunk rather than shifting towards the head, hindquarters or whichever body region is most visually prominent.

Consistent torso-centre placement was particularly important because this landmark was subsequently used as the spatial reference coordinate for RFID anchor assignment and behavioural analyses.

### 5. Tail base

The tail-base node was defined as the anatomical point at which the tail attaches to the body.

The node should not be positioned according to the first clearly visible tail pixel or changes in fur appearance.

---

## Occlusion rules

### Nose occluded

If the anatomical position of the nose can be inferred with reasonable confidence from the visible body orientation, its position may be estimated. If its location cannot reasonably be inferred, the landmark should be marked as not visible.

### One ear occluded

The visible ear should be labelled normally. An occluded ear should be marked as not visible when its anatomical position cannot reasonably be inferred rather than assigning an unsupported position.

### Tail base occluded

If body orientation provides sufficient anatomical information to infer the tail-base position, its position may be estimated. Otherwise, it should be marked as not visible.

### Animal partially occluded by another animal

Landmarks should not be assigned solely by guesswork during physical overlap. If the anatomical position of a landmark cannot reasonably be inferred, it should be marked as not visible.

---

## Additional conditions

### Contact with the arena wall

Contact with the arena wall should not alter the anatomical definition of a landmark. Nodes should continue to represent their defined anatomical positions.

### Motion blur

Where a landmark remains identifiable but its boundary is blurred by movement, the node should be placed at the estimated centre of the corresponding anatomical structure.

---

## Annotation quality-control checklist

Before completing annotation of a frame, the following criteria should be checked:

- Nose corresponds to the snout tip.
- Left and right ears correspond to the animal's anatomical left and right sides.
- Torso centre is positioned consistently at the approximate centre of the trunk.
- Tail base corresponds to the anatomical attachment of the tail.
- Landmarks that cannot reasonably be inferred are marked as not visible rather than assigned unsupported positions.
- The resulting skeleton is anatomically plausible.
- All animals within the frame are annotated according to the same landmark definitions and visibility rules.
