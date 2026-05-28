#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rotation-invariant gesture classifier for the landmark UDP stream.

Logic
-----
Per finger, "up" = dist(wrist, fingertip) > dist(wrist, PIP joint).
Uses 3D Euclidean distance so wrist tilt cannot break detection.
Thumb is deliberately ignored — its anatomy makes the wrist-distance
test unreliable for stability.

State machine (strictly mutually exclusive — no overlap by construction)
------------------------------------------------------------------------
  ZOOM_IN   : all 4 main fingers UP   (open hand)
  ZOOM_OUT  : all 4 main fingers DOWN (closed fist)
  PAN       : index UP, middle + ring + pinky DOWN (pointing)
  IDLE      : any other combination (transition / unrecognised)

This module is intentionally tiny and free of side effects so it can be
unit-tested without MediaPipe / OpenCV imports.
"""
import math


# (tip_index, pip_index) for the 4 fingers we actually use
_FINGERS = (
    (8,  6),   # index
    (12, 10),  # middle
    (16, 14),  # ring
    (20, 18),  # pinky
)

# Tolerance: webcams distort 3D distance, so a fully-extended finger's tip
# may not always measure strictly farther than the PIP from the wrist.
# Multiplying the PIP distance by < 1.0 makes the test more forgiving.
# 0.9 = tip only needs to be ≥ 90% of the PIP distance to count as UP.
_EXTENSION_TOLERANCE = 0.9


def _dist3d(a, b) -> float:
    """3D Euclidean distance between two objects with .x .y .z attributes."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_finger_up(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """True iff dist(wrist→tip) > dist(wrist→PIP) × _EXTENSION_TOLERANCE.

    Rotation-invariant: works at any hand orientation because the comparison
    is between two scalar distances measured from the same anchor (wrist).
    The tolerance multiplier (< 1.0) compensates for webcam-induced
    distance distortion that previously rejected genuinely extended fingers.
    """
    wrist = landmarks[0]
    return (
        _dist3d(wrist, landmarks[tip_idx])
        > _dist3d(wrist, landmarks[pip_idx]) * _EXTENSION_TOLERANCE
    )


def classify(landmarks) -> str:
    """Classify the current hand pose into one of {ZOOM_IN, ZOOM_OUT, PAN, IDLE}.

    Parameters
    ----------
    landmarks : sequence of 21 objects each having .x .y .z attributes
                (MediaPipe's hand_landmarks.landmark works directly)

    Returns
    -------
    str : "ZOOM_IN" | "ZOOM_OUT" | "PAN" | "IDLE"
    """
    fingers_up = tuple(
        is_finger_up(landmarks, tip, pip) for tip, pip in _FINGERS
    )

    # All four UP → open hand → zoom in
    if all(fingers_up):
        return "ZOOM_IN"

    # All four DOWN → fist → zoom out
    if not any(fingers_up):
        return "ZOOM_OUT"

    # Index UP, every other finger DOWN → pointing → pan
    if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return "PAN"

    return "IDLE"
