#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Send gesture commands to Google Earth via mouse/keyboard (Windows).

Uses a background worker thread so mouse drag / keyboard events never block
the main camera loop.  Main thread calls tick() each frame (non-blocking);
the worker reads the latest active command and sends input at the right rate.
"""
import sys
import threading
import time

from utils.gesture_command import (
    CMD_CLOSE,
    CMD_NONE,
    CMD_PAN_DOWN,
    CMD_PAN_LEFT,
    CMD_PAN_RIGHT,
    CMD_PAN_UP,
    CMD_RESET_VIEW,
    CMD_ROTATE_LEFT,
    CMD_ROTATE_RIGHT,
    CMD_SELECT,
    CMD_STOP,
    CMD_ZOOM_IN,
    CMD_ZOOM_OUT,
)

try:
    from pynput.keyboard import Controller as KeyboardController, Key
    from pynput.mouse import Button, Controller as MouseController
except ImportError:
    KeyboardController = None
    Key = None
    Button = None
    MouseController = None

ONE_SHOT_COMMANDS = frozenset({CMD_RESET_VIEW, CMD_SELECT, CMD_STOP, CMD_CLOSE})
REPEATABLE_COMMANDS = frozenset({
    CMD_ZOOM_IN, CMD_ZOOM_OUT,
    CMD_PAN_UP, CMD_PAN_DOWN, CMD_PAN_LEFT, CMD_PAN_RIGHT,
    CMD_ROTATE_LEFT, CMD_ROTATE_RIGHT,
})


def _find_window_title_substring(title_substring="Google Earth"):
    """Return the first visible window handle whose title contains title_substring."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        user32 = ctypes.windll.user32
        matches = []

        def enum_proc(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if title_substring.lower() in buf.value.lower():
                        matches.append(hwnd)
            return True

        enum_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(enum_type(enum_proc), 0)
    except Exception:
        return None
    return matches[0] if matches else None


def _focus_window_title_substring(title_substring="Google Earth"):
    """Bring a visible window whose title contains *title_substring* to front."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        hwnd = _find_window_title_substring(title_substring)
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            return True
    except Exception:
        return False
    return False


def _window_rect_title_substring(title_substring="Google Earth"):
    """Return (left, top, right, bottom) for a matching visible window."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import ctypes.wintypes
        hwnd = _find_window_title_substring(title_substring)
        if not hwnd:
            return None
        rect = ctypes.wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return None
    return None


def arrange_google_earth_window(title_substring="Google Earth", left_offset=700):
    """Place Google Earth on the right side of the primary screen.

    Returns the computed camera rectangle (x, y, width, height) or None.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = _find_window_title_substring(title_substring)
        if not hwnd:
            return None
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        camera_width = min(max(left_offset, 640), screen_width // 2)
        earth_width = max(screen_width - camera_width, 640)
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.MoveWindow(hwnd, camera_width, 0, earth_width, screen_height, True)
        return (0, 0, camera_width, screen_height)
    except Exception:
        return None


class EarthBridge:
    """Maps stable gesture commands to Google Earth input via a background worker thread.

    tick() is non-blocking — it only updates the active command.
    The worker thread handles all mouse/keyboard I/O including sleep-based drag timing.
    """

    def __init__(self,
                 interval_ms=50,
                 auto_focus=False,
                 window_title="Google Earth",
                 allow_close=False,
                 click_viewport=True,
                 pan_taps=5,
                 idle_grace_ms=450,
                 use_mouse_navigation=True,
                 pan_drag_px=130,
                 horizontal_pan_drag_px=50,
                 zoom_scroll_steps=4):
        if KeyboardController is None or MouseController is None:
            raise ImportError(
                "pynput is required for --control-earth. "
                "Install with: pip install pynput"
            )
        self.interval_sec = interval_ms / 1000.0
        self.auto_focus = auto_focus
        self.window_title = window_title
        self.allow_close = allow_close
        self.click_viewport = click_viewport
        self.pan_taps = max(1, pan_taps)
        self.idle_grace_sec = idle_grace_ms / 1000.0
        self.use_mouse_navigation = use_mouse_navigation
        self.pan_drag_px = max(20, pan_drag_px)
        self.horizontal_pan_drag_px = max(20, horizontal_pan_drag_px)
        self.zoom_scroll_steps = max(1, zoom_scroll_steps)

        self._keyboard = KeyboardController()
        self._mouse = MouseController()

        # Shared active command — main thread writes, worker reads
        self._active_command = CMD_NONE
        self._cmd_lock = threading.Lock()

        # Idle grace tracking (main thread only)
        self._last_active_time = 0.0

        # Status string — worker writes, main thread reads (GIL-safe for str)
        self.last_status = "idle"

        # Background worker
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="EarthBridgeWorker"
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Public API (main thread)
    # ------------------------------------------------------------------

    def tick(self, command):
        """Update active command — non-blocking, safe to call every frame."""
        now = time.monotonic()

        if command == CMD_NONE:
            with self._cmd_lock:
                self._active_command = CMD_NONE
            if now - self._last_active_time <= self.idle_grace_sec:
                self.last_status = "waiting"
            else:
                self.last_status = "idle"
            return False

        if command == CMD_CLOSE and not self.allow_close:
            with self._cmd_lock:
                self._active_command = CMD_NONE
            self.last_status = "blocked"
            return False

        self._last_active_time = now
        with self._cmd_lock:
            self._active_command = command
        return False

    # ------------------------------------------------------------------
    # Low-level input helpers (called from worker thread only)
    # ------------------------------------------------------------------

    def _viewport_point(self, x_ratio=0.64, y_ratio=0.52):
        rect = _window_rect_title_substring(self.window_title)
        if rect is None:
            return None
        left, top, right, bottom = rect
        width = max(1, right - left)
        height = max(1, bottom - top)
        return (int(left + width * x_ratio), int(top + height * y_ratio))

    def _do_focus(self, click_vp=False):
        """Focus Earth window; optionally click into the 3D viewport."""
        if not self.auto_focus:
            return
        _focus_window_title_substring(self.window_title)
        if not (self.click_viewport and click_vp):
            return
        point = self._viewport_point()
        if point is None:
            return
        prev = self._mouse.position
        self._mouse.position = point
        self._mouse.click(Button.left, 1)
        self._mouse.position = prev

    def _tap(self, key):
        self._keyboard.press(key)
        self._keyboard.release(key)

    def _tap_char(self, char):
        self._keyboard.press(char)
        self._keyboard.release(char)

    def _tap_shift_arrow(self, arrow_key):
        with self._keyboard.pressed(Key.shift):
            self._tap(arrow_key)

    def _tap_many(self, key, count):
        for _ in range(count):
            self._tap(key)

    def _scroll_viewport(self, steps):
        point = self._viewport_point()
        if point is None:
            return False
        prev = self._mouse.position
        self._mouse.position = point
        self._mouse.scroll(0, steps)
        self._mouse.position = prev
        return True

    def _drag_viewport(self, dx, dy):
        """Drag inside the Earth viewport. Sleeps are safe here (worker thread)."""
        point = self._viewport_point()
        if point is None:
            return False
        x, y = point
        prev = self._mouse.position
        self._mouse.position = (x, y)
        self._mouse.press(Button.left)
        time.sleep(0.01)
        self._mouse.position = (x + dx // 2, y + dy // 2)
        time.sleep(0.01)
        self._mouse.position = (x + dx, y + dy)
        time.sleep(0.01)
        self._mouse.release(Button.left)
        self._mouse.position = prev
        return True

    def _send_mouse_for_command(self, command):
        if command == CMD_ZOOM_IN:
            return self._scroll_viewport(self.zoom_scroll_steps)
        if command == CMD_ZOOM_OUT:
            return self._scroll_viewport(-self.zoom_scroll_steps)
        if command == CMD_PAN_UP:
            return self._drag_viewport(0, self.pan_drag_px)
        if command == CMD_PAN_DOWN:
            return self._drag_viewport(0, -self.pan_drag_px)
        if command == CMD_PAN_LEFT:
            return self._drag_viewport(self.horizontal_pan_drag_px, 0)
        if command == CMD_PAN_RIGHT:
            return self._drag_viewport(-self.horizontal_pan_drag_px, 0)
        return False

    def _send_keys_for_command(self, command):
        if command == CMD_ZOOM_IN:
            self._tap(Key.page_up)
        elif command == CMD_ZOOM_OUT:
            self._tap(Key.page_down)
        elif command == CMD_PAN_UP:
            self._tap_many(Key.up, self.pan_taps)
        elif command == CMD_PAN_DOWN:
            self._tap_many(Key.down, self.pan_taps)
        elif command == CMD_PAN_LEFT:
            self._tap_many(Key.left, self.pan_taps)
        elif command == CMD_PAN_RIGHT:
            self._tap_many(Key.right, self.pan_taps)
        elif command == CMD_ROTATE_LEFT:
            self._tap_shift_arrow(Key.left)
        elif command == CMD_ROTATE_RIGHT:
            self._tap_shift_arrow(Key.right)
        elif command == CMD_RESET_VIEW:
            self._tap_char("r")
        elif command == CMD_SELECT:
            self._tap(Key.enter)
        elif command == CMD_STOP:
            self._tap(Key.space)
        elif command == CMD_CLOSE:
            if self.allow_close:
                with self._keyboard.pressed(Key.alt):
                    self._tap(Key.f4)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Run forever in a daemon thread; sends input without blocking main loop."""
        last_sent = CMD_NONE
        last_send_t = 0.0
        last_focus_t = 0.0

        while True:
            with self._cmd_lock:
                cmd = self._active_command

            if cmd == CMD_NONE:
                last_sent = CMD_NONE
                time.sleep(0.010)
                continue

            now = time.monotonic()
            changed = cmd != last_sent
            elapsed = now - last_send_t

            # Re-focus viewport when command changes or every 2 s
            if changed or now - last_focus_t >= 2.0:
                self._do_focus(click_vp=changed)
                last_focus_t = now

            should_send = False
            if cmd in ONE_SHOT_COMMANDS:
                should_send = changed
            elif cmd in REPEATABLE_COMMANDS:
                should_send = changed or elapsed >= self.interval_sec

            if should_send:
                try:
                    if cmd in REPEATABLE_COMMANDS and self.use_mouse_navigation:
                        if not self._send_mouse_for_command(cmd):
                            self._send_keys_for_command(cmd)
                        self.last_status = "driving"
                    else:
                        self._send_keys_for_command(cmd)
                        self.last_status = "stopped" if cmd == CMD_STOP else "applied"
                except Exception:
                    pass
                last_sent = cmd
                last_send_t = now
            else:
                self.last_status = "driving" if cmd in REPEATABLE_COMMANDS else "waiting"

            time.sleep(0.005)  # ~200 Hz loop; avoids busy-wait
