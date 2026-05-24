#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Map hand pose + finger motion to navigation control commands."""
from collections import Counter, deque

# Control commands
CMD_NONE = "NONE"
CMD_ZOOM_IN = "ZOOM_IN"
CMD_ZOOM_OUT = "ZOOM_OUT"
CMD_PAN_UP = "PAN_UP"
CMD_PAN_DOWN = "PAN_DOWN"
CMD_PAN_LEFT = "PAN_LEFT"
CMD_PAN_RIGHT = "PAN_RIGHT"
CMD_ROTATE_LEFT = "ROTATE_LEFT"
CMD_ROTATE_RIGHT = "ROTATE_RIGHT"
CMD_RESET_VIEW = "RESET_VIEW"
CMD_SELECT = "SELECT"
CMD_STOP = "STOP"
CMD_CLOSE = "CLOSE"

# UX states for the digital twin state machine
UX_IDLE = "IDLE"
UX_NAVIGATING = "NAVIGATING"
UX_ZOOMING = "ZOOMING"
UX_STOP_PENDING = "STOP_PENDING"
UX_STOPPED = "STOPPED"
UX_RECENTERING = "RECENTERING"

_PAN_COMMANDS = frozenset({CMD_PAN_UP, CMD_PAN_DOWN, CMD_PAN_LEFT, CMD_PAN_RIGHT})
_ZOOM_COMMANDS = frozenset({CMD_ZOOM_IN, CMD_ZOOM_OUT})

GE_ACTIONS = {
    CMD_ZOOM_IN: "Zoom in",
    CMD_ZOOM_OUT: "Zoom out",
    CMD_PAN_UP: "Pan north",
    CMD_PAN_DOWN: "Pan south",
    CMD_PAN_LEFT: "Pan west",
    CMD_PAN_RIGHT: "Pan east",
    CMD_ROTATE_LEFT: "Rotate left",
    CMD_ROTATE_RIGHT: "Rotate right",
    CMD_RESET_VIEW: "Reset view",
    CMD_SELECT: "Select / click",
    CMD_STOP: "Stop navigation",
    CMD_CLOSE: "Exit application",
    CMD_NONE: "Idle",
}


def bbox_area(brect):
    return max(0, brect[2] - brect[0]) * max(0, brect[3] - brect[1])


def hand_center(brect):
    return ((brect[0] + brect[2]) / 2.0, (brect[1] + brect[3]) / 2.0)


def _pan_from_avg_halves(points, min_delta):
    """Compute pan direction by comparing average of first half vs second half."""
    if len(points) < 4:
        return CMD_NONE
    mid = len(points) // 2
    x0 = sum(p[0] for p in points[:mid]) / mid
    y0 = sum(p[1] for p in points[:mid]) / mid
    n2 = len(points) - mid
    x1 = sum(p[0] for p in points[mid:]) / n2
    y1 = sum(p[1] for p in points[mid:]) / n2
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < min_delta and abs(dy) < min_delta:
        return CMD_NONE
    if abs(dx) >= abs(dy):
        return CMD_PAN_RIGHT if dx > 0 else CMD_PAN_LEFT
    return CMD_PAN_DOWN if dy > 0 else CMD_PAN_UP


def pan_from_point_history(point_history, min_delta=22):
    """Pan direction from index-finger-tip trail (Pointer gesture)."""
    valid = [p for p in point_history if p[0] != 0 and p[1] != 0]
    if len(valid) < 6:
        return CMD_NONE
    return _pan_from_avg_halves(valid, min_delta)


def pan_from_hand_movement(history, min_delta=18):
    """Pan direction from hand-center movement (Open gesture)."""
    pts = list(history)
    if len(pts) < 4:
        return CMD_NONE
    return _pan_from_avg_halves(pts, min_delta)


def finger_spread_score(landmark_list, brect):
    """Measure how open the hand is using fingertip distance from palm center."""
    if landmark_list is None or len(landmark_list) < 21:
        return None

    palm_indices = (0, 5, 9, 13, 17)
    tip_indices = (4, 8, 12, 16, 20)
    palm_x = sum(landmark_list[i][0] for i in palm_indices) / len(palm_indices)
    palm_y = sum(landmark_list[i][1] for i in palm_indices) / len(palm_indices)

    bbox_w = max(1, brect[2] - brect[0])
    bbox_h = max(1, brect[3] - brect[1])
    bbox_diag = (bbox_w ** 2 + bbox_h ** 2) ** 0.5

    distances = []
    for index in tip_indices:
        dx = landmark_list[index][0] - palm_x
        dy = landmark_list[index][1] - palm_y
        distances.append((dx ** 2 + dy ** 2) ** 0.5)

    return (sum(distances) / len(distances)) / bbox_diag


def distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx ** 2 + dy ** 2) ** 0.5


def count_extended_fingers(landmark_list):
    """Return an approximate count of extended fingers from 2D landmarks."""
    if landmark_list is None or len(landmark_list) < 21:
        return 0

    extended = 0
    # Index, middle, ring, pinky: fingertip above PIP joint means extended.
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if landmark_list[tip][1] < landmark_list[pip][1]:
            extended += 1

    # Thumb: compare horizontal distance from palm center as a loose signal.
    palm_x = landmark_list[0][0]
    thumb_tip_x = landmark_list[4][0]
    index_mcp_x = landmark_list[5][0]
    if abs(thumb_tip_x - palm_x) > abs(index_mcp_x - palm_x) * 0.75:
        extended += 1

    return extended


def is_zero_gesture(landmark_list, brect):
    """Detect a simple OK/zero gesture: thumb and index tips are close."""
    if landmark_list is None or len(landmark_list) < 21:
        return False
    bbox_w = max(1, brect[2] - brect[0])
    bbox_h = max(1, brect[3] - brect[1])
    bbox_diag = (bbox_w ** 2 + bbox_h ** 2) ** 0.5
    thumb_index = distance(landmark_list[4], landmark_list[8])
    return thumb_index / bbox_diag < 0.18


def simple_readme_command(landmark_list, brect):
    """
    Map simple static gestures to zoom controls.

    Pan is handled later from hand position/movement so an open hand can travel
    around the map instead of always meaning one fixed direction.
    """
    if landmark_list is None or len(landmark_list) < 21:
        return CMD_NONE, "simple_no_landmarks"

    if is_zero_gesture(landmark_list, brect):
        return CMD_ZOOM_OUT, "simple_zero_zoom_out"

    extended = count_extended_fingers(landmark_list)
    if extended <= 1:
        return CMD_ZOOM_IN, "simple_closed_zoom_in"

    return CMD_NONE, "simple_idle"


class GestureCommandMapper:
    """Raw per-frame command from pose + motion classifiers with UX state machine."""

    def __init__(self,
                 zoom_ratio_threshold=0.035,
                 spread_ratio_threshold=0.06,
                 close_frames_for_exit=45,
                 stop_frames_for_stop=8,
                 nav_flicker_threshold=6):
        self.zoom_ratio_threshold = zoom_ratio_threshold
        self.spread_ratio_threshold = spread_ratio_threshold
        self.close_frames_for_exit = close_frames_for_exit
        self.stop_frames_for_stop = stop_frames_for_stop
        self.nav_flicker_threshold = nav_flicker_threshold
        self.bbox_area_history = deque(maxlen=8)
        self.spread_history = deque(maxlen=8)
        self.open_hand_center_history = deque(maxlen=10)
        self.close_hold_frames = 0
        self.stop_hold_frames = 0
        self.force_reset = False
        self.ux_state = UX_IDLE
        self.nav_flicker_frames = 0

    def request_reset(self):
        self.force_reset = True

    def reset_hand_tracking(self):
        """Call when no hand is visible in the frame."""
        self.close_hold_frames = 0
        self.stop_hold_frames = 0
        self.nav_flicker_frames = 0
        self.bbox_area_history.clear()
        self.spread_history.clear()
        self.open_hand_center_history.clear()
        if self.ux_state not in (UX_STOPPED, UX_RECENTERING):
            self.ux_state = UX_IDLE

    def map_command(self, hand_sign, finger_gesture, brect, image_shape,
                    point_history, landmark_list=None):
        # Reset is always immediate regardless of current state
        if self.force_reset:
            self.force_reset = False
            self.ux_state = UX_RECENTERING
            self.nav_flicker_frames = 0
            self.stop_hold_frames = 0
            return CMD_RESET_VIEW, "keyboard_reset"

        # RECENTERING lasts one frame then returns to IDLE
        if self.ux_state == UX_RECENTERING:
            self.ux_state = UX_IDLE

        image_height, image_width = image_shape[0], image_shape[1]
        area = bbox_area(brect)
        self.bbox_area_history.append(area)
        spread_score = finger_spread_score(landmark_list, brect)
        if spread_score is not None:
            self.spread_history.append(spread_score)

        # Clear open-hand movement history when gesture is not Open
        if hand_sign != "Open":
            self.open_hand_center_history.clear()

        # Compute raw intent from gesture classifiers
        raw_cmd = CMD_NONE
        raw_source = "idle"

        # Pointer/Open have few extended fingers — skip zoom detection to avoid conflict
        if hand_sign in ("Pointer", "Open"):
            simple_command, simple_source = CMD_NONE, "gesture_skip_simple"
        else:
            simple_command, simple_source = simple_readme_command(landmark_list, brect)
        if simple_command != CMD_NONE:
            raw_cmd, raw_source = simple_command, simple_source
        elif hand_sign == "OK":
            self.close_hold_frames = 0
            raw_cmd, raw_source = CMD_ZOOM_OUT, "hand_sign_OK_zoom_out"
        elif hand_sign == "Close":
            self.close_hold_frames += 1
            raw_cmd, raw_source = CMD_ZOOM_IN, "hand_sign_Close_zoom_in"
        else:
            self.close_hold_frames = 0
            if finger_gesture == "Clockwise":
                raw_cmd, raw_source = CMD_ROTATE_RIGHT, "finger_Clockwise"
            elif finger_gesture == "Counter Clockwise":
                raw_cmd, raw_source = CMD_ROTATE_LEFT, "finger_Counter_Clockwise"
            elif hand_sign == "Open":
                # Movement-based pan: swipe open hand in a direction to pan that way
                cx, cy = hand_center(brect)
                self.open_hand_center_history.append((cx, cy))
                pan_cmd = pan_from_hand_movement(self.open_hand_center_history)
                if pan_cmd != CMD_NONE:
                    raw_cmd, raw_source = pan_cmd, "open_hand_move"
            elif hand_sign == "Pointer" and finger_gesture == "Move":
                pan_cmd = pan_from_point_history(point_history)
                if pan_cmd != CMD_NONE:
                    raw_cmd, raw_source = pan_cmd, "pointer_move_trail"
            elif finger_gesture == "Stop":
                raw_cmd, raw_source = CMD_STOP, "finger_Stop_raw"

        return self._apply_state(raw_cmd, raw_source)

    def _apply_state(self, raw_cmd, raw_source):
        """Apply state machine: stop hold threshold and navigation flicker suppression."""
        # Stop gesture requires sustained hold to produce CMD_STOP
        if raw_cmd == CMD_STOP:
            self.stop_hold_frames += 1
            if self.stop_hold_frames < self.stop_frames_for_stop:
                # Suppress stop flicker entirely during active pan navigation
                if self.ux_state == UX_NAVIGATING:
                    return CMD_NONE, "nav_stop_suppressed"
                self.ux_state = UX_STOP_PENDING
                return CMD_NONE, "finger_Stop_pending"
            # Hold threshold reached: commit to stopped
            self.ux_state = UX_STOPPED
            self.nav_flicker_frames = 0
            return CMD_STOP, "finger_Stop_hold"

        self.stop_hold_frames = 0

        # Non-stop gesture cancels STOP_PENDING
        if self.ux_state == UX_STOP_PENDING:
            self.ux_state = UX_IDLE

        # In STOPPED, require a new intentional gesture to resume
        if self.ux_state == UX_STOPPED:
            if raw_cmd == CMD_NONE:
                return CMD_NONE, "stopped_idle"
            self.ux_state = UX_IDLE

        # During NAVIGATING, suppress non-pan gestures for nav_flicker_threshold frames
        if self.ux_state == UX_NAVIGATING:
            if raw_cmd in _PAN_COMMANDS:
                self.nav_flicker_frames = 0
                return raw_cmd, raw_source
            self.nav_flicker_frames += 1
            if self.nav_flicker_frames < self.nav_flicker_threshold:
                return CMD_NONE, "nav_flicker_suppressed"
            # Enough consecutive non-pan frames: exit NAVIGATING
            self.nav_flicker_frames = 0
            self.ux_state = UX_IDLE

        # Normal state transitions
        if raw_cmd in _PAN_COMMANDS:
            self.ux_state = UX_NAVIGATING
            self.nav_flicker_frames = 0
        elif raw_cmd in _ZOOM_COMMANDS:
            self.ux_state = UX_ZOOMING
        elif raw_cmd == CMD_NONE and self.ux_state not in (UX_STOPPED, UX_STOP_PENDING, UX_RECENTERING):
            self.ux_state = UX_IDLE

        return raw_cmd, raw_source


class CommandStabilizer:
    """Majority vote over recent frames to reduce flicker."""

    def __init__(self, buffer_len=10, min_votes=6, hold_frames=12):
        self.buffer = deque(maxlen=buffer_len)
        self.min_votes = min(min_votes, buffer_len)
        self.hold_frames = hold_frames
        self.last_active_command = CMD_NONE
        self.last_active_confidence = 0.0
        self.idle_frames = 0

    def update(self, command):
        self.buffer.append(command)
        if not self.buffer:
            return CMD_NONE, 0.0
        counts = Counter(self.buffer)
        cmd, votes = counts.most_common(1)[0]
        confidence = votes / len(self.buffer)
        if votes >= self.min_votes and cmd != CMD_NONE:
            self.last_active_command = cmd
            self.last_active_confidence = confidence
            self.idle_frames = 0
            return cmd, confidence
        if cmd == CMD_NONE and votes >= self.min_votes:
            self.idle_frames += 1
            if (self.last_active_command != CMD_NONE and
                    self.idle_frames <= self.hold_frames):
                return self.last_active_command, max(
                    0.35,
                    self.last_active_confidence * 0.75,
                )
            self.last_active_command = CMD_NONE
            self.last_active_confidence = confidence
            return CMD_NONE, confidence
        if self.last_active_command != CMD_NONE:
            self.idle_frames += 1
            if self.idle_frames <= self.hold_frames:
                return self.last_active_command, max(
                    0.35,
                    self.last_active_confidence * 0.6,
                )
        return CMD_NONE, confidence


def command_payload(command, confidence, source, hand_sign="", finger_gesture="",
                    ux_state=UX_IDLE):
    return {
        "command": command,
        "confidence": round(confidence, 2),
        "source": source,
        "hand_sign": hand_sign,
        "finger_gesture": finger_gesture,
        "earth_action": GE_ACTIONS.get(command, ""),
        "ux_state": ux_state,
    }
