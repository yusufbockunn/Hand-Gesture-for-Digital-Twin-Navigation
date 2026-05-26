#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Send gesture command payloads to a Unity application via UDP.

Usage
-----
    bridge = UnityBridge(host="127.0.0.1", port=7777)
    bridge.tick(active_payload)   # call every frame — non-blocking
    bridge.close()                # call on app exit

Unity-side counterpart: unity/GestureReceiver.cs
"""
import json
import socket

from utils.gesture_command import CMD_NONE


class UnityBridge:
    """Fire-and-forget UDP sender for gesture payloads.

    Non-blocking: socket.sendto() completes in microseconds and never retries.
    Packet loss is acceptable — the next frame will carry an updated command.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7777) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_status = "idle"

    def tick(self, payload: dict) -> None:
        """Serialize *payload* to JSON and send to Unity. Non-blocking."""
        command = payload.get("command", CMD_NONE)
        try:
            data = json.dumps(payload).encode("utf-8")
            self._sock.sendto(data, self._addr)
            self.last_status = "idle" if command == CMD_NONE else "sent"
        except OSError:
            self.last_status = "error"

    def close(self) -> None:
        self._sock.close()
