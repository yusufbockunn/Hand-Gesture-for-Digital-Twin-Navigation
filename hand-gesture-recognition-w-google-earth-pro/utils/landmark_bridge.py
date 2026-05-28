#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stream hand landmarks + classified command + index-tip position to Unity.

Packet format (JSON, sent every frame)
--------------------------------------
  Hand detected:
    {
      "command":  "ZOOM_IN" | "ZOOM_OUT" | "PAN" | "IDLE",
      "index_x":  0.0–1.0   (MediaPipe normalised X of landmark 8),
      "index_y":  0.0–1.0   (MediaPipe normalised Y of landmark 8),
      "landmarks": [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
    }

  No hand:
    {"command":"IDLE", "index_x":0.0, "index_y":0.0, "landmarks":[]}

Unity-side counterpart: Assets/Scripts/HandTracking/HandLandmarkReceiver.cs
"""
import json
import socket


class LandmarkBridge:
    """Fire-and-forget UDP sender (port 7778 by default).

    Non-blocking: sendto() completes in microseconds and never retries.
    Packet loss is acceptable — the next frame carries fresh data.
    Flat float array for landmarks so Unity's built-in JsonUtility
    can parse it without Newtonsoft.Json.
    """

    _NO_HAND_PACKET = json.dumps({
        "command":  "IDLE",
        "index_x":  0.0,
        "index_y":  0.0,
        "landmarks": [],
    }).encode("utf-8")

    def __init__(self, host: str = "127.0.0.1", port: int = 7778) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(
        self,
        landmarks,
        command: str = "IDLE",
        index_x: float = 0.0,
        index_y: float = 0.0,
    ) -> None:
        """Send a per-frame packet. Non-blocking.

        Parameters
        ----------
        landmarks : iterable of (float, float, float)
            21 (x, y, z) tuples in MediaPipe normalised coordinates.
        command : str
            Classifier output: "ZOOM_IN", "ZOOM_OUT", "PAN", or "IDLE".
        index_x, index_y : float
            MediaPipe normalised position of INDEX_TIP (landmark 8).
            Unity uses these directly for pan deltas — no on-receiver math.
        """
        flat = [round(v, 5) for xyz in landmarks for v in xyz]
        payload = json.dumps({
            "command":  command,
            "index_x":  round(float(index_x), 5),
            "index_y":  round(float(index_y), 5),
            "landmarks": flat,
        }).encode("utf-8")
        try:
            self._sock.sendto(payload, self._addr)
        except OSError:
            pass

    def send_no_hand(self) -> None:
        """Notify Unity that no hand is currently visible. Non-blocking."""
        try:
            self._sock.sendto(self._NO_HAND_PACKET, self._addr)
        except OSError:
            pass

    def close(self) -> None:
        """Release the UDP socket. Call once on application exit."""
        self._sock.close()
