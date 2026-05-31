#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analog gesture input pipeline for spatial / 6-DoF navigation.

Architecture
------------
  MediaPipe landmarks
        │
        ▼
  AnalogGestureMapper   ← applies OneEuroFilter to raw coordinates
        │                  outputs 6 continuous axis values in [-1, 1]
        ▼
  VirtualControllerBridge  ← writes axes to a vJoy virtual joystick
        │
        ▼
  Google Earth Pro (or any DirectInput-aware app)

Prerequisites
-------------
  1. vJoy driver (v2.x):  https://github.com/shauleiz/vJoy/releases
     Configure device 1 with axes: X, Y, Z, Rx, Ry, Rz  (min 6 axes)
  2. pip install pyvjoy
  3. Google Earth Pro → Tools → Options → Navigation → enable controller

Notes
-----
- All public methods are non-blocking; no time.sleep() anywhere in this module.
- For Google Earth *Web* (browser), vJoy has no effect — the Web version does
  not expose a joystick API.  Use EarthBridge (mouse simulation) for the Web.
- Google Earth Pro's native 6-DoF support is best when the device presents
  itself as a SpaceNavigator.  vJoy appears as a generic DirectInput device,
  which GE Pro maps to navigation but with less fidelity than SpaceNavigator.
  If you need full SpaceNavigator emulation, look at the 3DxWare SDK.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple


__all__ = [
    "OneEuroFilter",
    "OneEuroFilterVec",
    "VirtualControllerBridge",
    "AnalogGestureMapper",
    "Axes6DoF",
]


# ══════════════════════════════════════════════════════════════════════
# §1  1 Euro Filter  (Casiez, Roussel, Vogel — CHI 2012)
# ══════════════════════════════════════════════════════════════════════

class _LowPassFilter:
    """Single-pole IIR low-pass filter (internal building block)."""

    __slots__ = ("_y", "_alpha")

    def __init__(self) -> None:
        self._y: float | None = None
        self._alpha: float = 1.0

    def set_alpha(self, alpha: float) -> None:
        self._alpha = max(0.0, min(1.0, alpha))

    def __call__(self, x: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = self._alpha * x + (1.0 - self._alpha) * self._y
        return self._y  # type: ignore[return-value]

    @property
    def last(self) -> float:
        return self._y if self._y is not None else 0.0

    def reset(self) -> None:
        self._y = None


class OneEuroFilter:
    """
    1 Euro Filter — adaptive single-axis low-pass filter.

    Key property
    ~~~~~~~~~~~~
    - Signal **still**  → cutoff drops  → heavy smoothing  → jitter ≈ 0
    - Signal **moving** → cutoff rises  → lighter smoothing → lag   ≈ 0

    Parameters
    ----------
    freq        : Sampling rate in Hz (updated automatically when timestamps given).
    min_cutoff  : Minimum cutoff frequency (Hz).
                  Lower  → smoother at rest, but more lag when starting to move.
                  Typical range: 0.5 – 3.0
    beta        : Speed coefficient.
                  Higher → less lag for fast movements, but more noise.
                  Typical range: 0.001 – 0.5
    d_cutoff    : Cutoff for the derivative pre-filter (1 Hz is almost always fine).

    Usage
    -----
        f = OneEuroFilter(freq=30, min_cutoff=1.5, beta=0.05)
        for raw_val, t in stream:
            smooth = f(raw_val, timestamp=t)
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.5,
                 beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self._freq = max(1e-6, freq)
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filt = _LowPassFilter()
        self._dx_filt = _LowPassFilter()
        self._t_prev: float | None = None

    # ------------------------------------------------------------------

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        """IIR alpha from cutoff (Hz) and sampling rate (Hz)."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te  = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """
        Filter one sample.

        Parameters
        ----------
        x         : Raw measurement (any unit).
        timestamp : time.monotonic() in seconds — enables adaptive frequency.
                    Pass None to use the fixed freq set in __init__.

        Returns
        -------
        Filtered value in the same unit as x.
        """
        # Update frequency estimate from inter-sample time
        if timestamp is not None and self._t_prev is not None:
            dt = timestamp - self._t_prev
            if dt > 1e-6:
                self._freq = 1.0 / dt
        if timestamp is not None:
            self._t_prev = timestamp

        # Step 1 — estimate and smooth derivative
        dx_raw = (x - self._x_filt.last) * self._freq
        self._dx_filt.set_alpha(self._alpha(self._d_cutoff, self._freq))
        dx_hat = self._dx_filt(dx_raw)

        # Step 2 — adaptive cutoff scales with signal speed
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)

        # Step 3 — filter signal
        self._x_filt.set_alpha(self._alpha(cutoff, self._freq))
        return self._x_filt(x)

    def reset(self) -> None:
        """Clear state (e.g. when the tracked object disappears)."""
        self._x_filt.reset()
        self._dx_filt.reset()
        self._t_prev = None


class OneEuroFilterVec:
    """
    Convenience wrapper: apply independent OneEuroFilters to each dimension
    of a 2-D or 3-D coordinate tuple.

    Usage
    -----
        f2d = OneEuroFilterVec(dims=2, min_cutoff=1.5, beta=0.05)
        smooth_xy = f2d((raw_x, raw_y), timestamp=t)
    """

    def __init__(self, dims: int = 2, **kwargs) -> None:
        self._filters = [OneEuroFilter(**kwargs) for _ in range(dims)]

    def __call__(self, vec: tuple, timestamp: float | None = None) -> tuple:
        return tuple(f(v, timestamp) for f, v in zip(self._filters, vec))

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


# ══════════════════════════════════════════════════════════════════════
# §2  Virtual Controller Bridge
# ══════════════════════════════════════════════════════════════════════

try:
    import pyvjoy as _pyvjoy          # type: ignore[import]
    _PYVJOY_OK = True
except ImportError:
    _pyvjoy = None                    # type: ignore[assignment]
    _PYVJOY_OK = False


class VirtualControllerBridge:
    """
    6-DoF virtual joystick via vJoy + pyvjoy.

    Axis convention (SpaceNavigator-compatible)
    -------------------------------------------
      X  — pan  left  (–1) / right (+1)
      Y  — pan  up    (–1) / down  (+1)    [screen Y is inverted from world up]
      Z  — zoom in    (–1) / out   (+1)    [throttle / elevation axis]
      Rx — tilt  backward (–1) / forward (+1)
      Ry — roll  left    (–1) / right   (+1)
      Rz — yaw   left    (–1) / right   (+1)

    All setter methods accept values in [-1.0, +1.0].

    Dry-run mode
    ------------
    If pyvjoy is not installed or vJoy is not found, the class operates in
    dry-run mode: no exception is raised, axes are just silently discarded.
    Use `.available` to check.

    Example
    -------
        bridge = VirtualControllerBridge(device_id=1)
        if bridge.available:
            bridge.set_axes(x=-0.3, z=0.5)   # pan left, zoom out
        bridge.shutdown()
    """

    _AXIS_MIN = 0x0001
    _AXIS_MAX = 0x8000
    _AXIS_MID = (_AXIS_MIN + _AXIS_MAX) // 2   # 16384 — neutral center

    def __init__(self, device_id: int = 1) -> None:
        self._device_id = device_id
        self._dev = None
        self._available = False

        if not _PYVJOY_OK:
            print("[VirtualController] pyvjoy not installed — dry-run mode.")
            return

        try:
            self._dev = _pyvjoy.VJoyDevice(device_id)
            self.reset()
            self._available = True
            print(f"[VirtualController] vJoy device {device_id} ready.")
        except Exception as exc:
            print(f"[VirtualController] vJoy init failed: {exc}  — dry-run mode.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when a real vJoy device was acquired successfully."""
        return self._available

    def set_axes(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> None:
        """Write all 6 axes at once. Values are clamped to [-1, 1]."""
        if not self._available:
            return
        try:
            self._dev.set_axis(_pyvjoy.HID_USAGE_X,  self._encode(x))
            self._dev.set_axis(_pyvjoy.HID_USAGE_Y,  self._encode(y))
            self._dev.set_axis(_pyvjoy.HID_USAGE_Z,  self._encode(z))
            self._dev.set_axis(_pyvjoy.HID_USAGE_RX, self._encode(rx))
            self._dev.set_axis(_pyvjoy.HID_USAGE_RY, self._encode(ry))
            self._dev.set_axis(_pyvjoy.HID_USAGE_RZ, self._encode(rz))
        except Exception:
            pass

    def set_from_axes6dof(self, axes: "Axes6DoF") -> None:
        """Convenience: unpack an Axes6DoF named tuple directly."""
        self.set_axes(axes.x, axes.y, axes.z, axes.rx, axes.ry, axes.rz)

    def reset(self) -> None:
        """Return all axes to neutral (0.0)."""
        self.set_axes()

    def shutdown(self) -> None:
        """Release device gracefully."""
        self.reset()
        self._dev = None
        self._available = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(self, value: float) -> int:
        """Map [-1.0, +1.0] → [AXIS_MIN, AXIS_MAX]."""
        v = max(-1.0, min(1.0, float(value)))
        return int(self._AXIS_MID + v * (self._AXIS_MID - self._AXIS_MIN))


# ══════════════════════════════════════════════════════════════════════
# §3  Analog Gesture Mapper
# ══════════════════════════════════════════════════════════════════════

class Axes6DoF(NamedTuple):
    """6-DoF output — all values in [-1.0, +1.0]."""
    x:  float = 0.0   # pan left/right
    y:  float = 0.0   # pan up/down
    z:  float = 0.0   # zoom in/out
    rx: float = 0.0   # tilt
    ry: float = 0.0   # roll
    rz: float = 0.0   # yaw/rotate


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _dead_zone(v: float, dz: float) -> float:
    """Apply symmetric dead zone then rescale to preserve full output range."""
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0.0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


def _exp_curve(v: float, exp: float) -> float:
    """Exponential response curve (preserves sign, raises |v| to exp)."""
    return math.copysign(abs(v) ** exp, v)


class AnalogGestureMapper:
    """
    Convert MediaPipe hand landmarks into continuous 6-DoF joystick values.

    Mapping
    -------
    Pan  X, Y  : Filtered palm-center position offset from a calibrated neutral.
                 Moving hand left/right/up/down → proportional pan.
    Zoom Z     : Pinch distance (thumb tip ↔ index tip) normalised by hand size.
                 Closed pinch → zoom in (z < 0);  spread → zoom out (z > 0).
    Rotate Rz  : Wrist→index-MCP vector angle vs. horizontal.
    Rx, Ry     : Currently 0.0 (reserved for future tilt gestures).

    Coordinate convention
    ---------------------
    Landmark list uses pixel coords [x, y] as returned by MediaPipe / app.py's
    calc_landmark_list().  21 points, indices 0-20.

    Usage
    -----
        mapper  = AnalogGestureMapper()
        bridge  = VirtualControllerBridge()

        # inside video loop:
        axes = mapper.compute(landmark_list, brect, image.shape,
                              timestamp=time.monotonic())
        bridge.set_from_axes6dof(axes)

        # when hand disappears:
        mapper.reset()
        bridge.reset()

    Parameters
    ----------
    freq, min_cutoff, beta
        Passed to each OneEuroFilter.  Tune beta first: raise it if pan still
        lags; lower it if you see noise/jitter during steady hold.
    pan_dead_zone
        Fraction of full range treated as "resting" — prevents drift when
        holding hand still in the neutral zone.  0.05 – 0.12 is typical.
    zoom_dead_zone
        Same concept for the zoom (pinch) axis.
    pan_exponent, zoom_exponent
        > 1  → fine control near center, larger movements near edges.
        1.0  → perfectly linear.
    pan_scale, zoom_scale
        Final output multipliers.  Keep ≤ 1.0 unless the application needs
        extra sensitivity.
    """

    # MediaPipe landmark indices used here
    _WRIST      = 0
    _THUMB_TIP  = 4
    _INDEX_MCP  = 5
    _INDEX_TIP  = 8
    _MIDDLE_MCP = 9
    _RING_MCP   = 13
    _PINKY_MCP  = 17

    # Pinch-to-hand-size ratio that means "neutral zoom" (neither in nor out)
    _PINCH_NEUTRAL = 0.35

    def __init__(self,
                 freq: float          = 30.0,
                 min_cutoff: float    = 1.5,
                 beta: float          = 0.05,
                 pan_dead_zone: float = 0.08,
                 zoom_dead_zone: float = 0.10,
                 pan_exponent: float  = 1.5,
                 zoom_exponent: float = 1.8,
                 pan_scale: float     = 1.0,
                 zoom_scale: float    = 1.0) -> None:

        kw = dict(freq=freq, min_cutoff=min_cutoff, beta=beta)
        self._f_pan_x = OneEuroFilter(**kw)
        self._f_pan_y = OneEuroFilter(**kw)
        self._f_zoom  = OneEuroFilter(**kw)
        self._f_rot_z = OneEuroFilter(**kw)

        self._pan_dz   = pan_dead_zone
        self._zoom_dz  = zoom_dead_zone
        self._pan_exp  = pan_exponent
        self._zoom_exp = zoom_exponent
        self._pan_sc   = pan_scale
        self._zoom_sc  = zoom_scale

        self._neutral_nx: float | None = None   # normalised [0,1]
        self._neutral_ny: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, landmark_list: list, image_shape: tuple) -> None:
        """
        Record the current palm position as the zero-pan neutral point.

        Call this once when the user holds their hand in the "resting" pose,
        or it is called automatically on the first frame of compute().
        """
        h, w = image_shape[:2]
        cx, cy = self._palm_center(landmark_list)
        self._neutral_nx = cx / w
        self._neutral_ny = cy / h

    def compute(self, landmark_list: list, brect: list,
                image_shape: tuple,
                timestamp: float | None = None) -> Axes6DoF:
        """
        Map one frame of hand landmarks to 6-DoF axis values.

        This is the hot path — keep it allocation-light.

        Parameters
        ----------
        landmark_list : list[list[int, int]]  — 21 pixel-space [x, y] points
        brect         : [x1, y1, x2, y2]      — hand bounding rect in pixels
        image_shape   : (height, width[, ch])
        timestamp     : time.monotonic() in seconds (improves filter accuracy)

        Returns
        -------
        Axes6DoF named tuple — all values in [-1.0, +1.0].
        """
        h, w = image_shape[:2]
        ts = timestamp if timestamp is not None else time.monotonic()

        # Auto-calibrate neutral on first frame
        if self._neutral_nx is None:
            self.calibrate(landmark_list, image_shape)

        # ── Pan  ──────────────────────────────────────────────────────
        cx, cy = self._palm_center(landmark_list)
        # Signed offset from neutral, normalised to ~[-1, 1]
        raw_x = _clamp((cx / w - self._neutral_nx) * 2.0)   # ×2: edge→±1
        raw_y = _clamp((cy / h - self._neutral_ny) * 2.0)

        fx = self._f_pan_x(raw_x, ts)
        fy = self._f_pan_y(raw_y, ts)

        pan_x = _clamp(_exp_curve(_dead_zone(fx, self._pan_dz), self._pan_exp)
                       * self._pan_sc)
        pan_y = _clamp(_exp_curve(_dead_zone(fy, self._pan_dz), self._pan_exp)
                       * self._pan_sc)

        # ── Zoom  ─────────────────────────────────────────────────────
        bbox_diag = math.sqrt(
            max(1, brect[2] - brect[0]) ** 2 +
            max(1, brect[3] - brect[1]) ** 2
        )
        thumb = landmark_list[self._THUMB_TIP]
        index = landmark_list[self._INDEX_TIP]
        pinch_px = math.sqrt(
            (thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2
        )
        pinch_ratio = pinch_px / bbox_diag  # typically [0.05, 0.70]

        # Negative = pinch closed = zoom in; positive = spread = zoom out
        raw_z = _clamp((self._PINCH_NEUTRAL - pinch_ratio) * 3.0)
        fz = self._f_zoom(raw_z, ts)
        zoom_z = _clamp(
            _exp_curve(_dead_zone(fz, self._zoom_dz), self._zoom_exp)
            * self._zoom_sc
        )

        # ── Rotation (yaw)  ──────────────────────────────────────────
        wx, wy   = landmark_list[self._WRIST]
        imx, imy = landmark_list[self._INDEX_MCP]
        angle    = math.atan2(imy - wy, imx - wx)   # rad, ±π
        raw_rz   = _clamp(angle / math.pi)
        rot_z    = self._f_rot_z(raw_rz, ts)

        return Axes6DoF(x=pan_x, y=pan_y, z=zoom_z, rx=0.0, ry=0.0, rz=rot_z)

    def reset(self) -> None:
        """
        Clear all filter states and neutral calibration.
        Call this whenever the hand disappears from the frame.
        """
        self._f_pan_x.reset()
        self._f_pan_y.reset()
        self._f_zoom.reset()
        self._f_rot_z.reset()
        self._neutral_nx = None
        self._neutral_ny = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _palm_center(self, lm: list) -> tuple[float, float]:
        """Average of 5 palm-base landmarks → stable centre of the hand."""
        indices = (self._WRIST, self._INDEX_MCP, self._MIDDLE_MCP,
                   self._RING_MCP, self._PINKY_MCP)
        xs = [lm[i][0] for i in indices]
        ys = [lm[i][1] for i in indices]
        return sum(xs) / len(xs), sum(ys) / len(ys)


# ══════════════════════════════════════════════════════════════════════
# Integration sketch (not run directly)
# ══════════════════════════════════════════════════════════════════════
#
# Replace the EarthBridge block in app.py with something like:
#
#   from utils.analog_input import AnalogGestureMapper, VirtualControllerBridge
#
#   mapper = AnalogGestureMapper(min_cutoff=1.5, beta=0.05)
#   bridge = VirtualControllerBridge(device_id=1)
#
#   # Inside the "hand detected" branch, after landmark_list is computed:
#   axes = mapper.compute(landmark_list, brect, debug_image.shape,
#                         timestamp=time.monotonic())
#   bridge.set_from_axes6dof(axes)
#
#   # Inside the "no hand" branch:
#   mapper.reset()
#   bridge.reset()
#
# The CommandStabilizer / GestureCommandMapper pipeline can remain in
# parallel for the discrete commands that still need digital logic
# (STOP, RESET_VIEW, CLOSE, SELECT).
