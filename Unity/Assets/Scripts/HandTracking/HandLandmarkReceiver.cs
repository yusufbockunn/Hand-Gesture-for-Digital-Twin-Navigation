// Assets/Scripts/HandTracking/HandLandmarkReceiver.cs
// Receives MediaPipe landmarks + classified command + index-tip viewport
// position from Python over UDP.
//
// Python launch:  python app.py --stream-landmarks --landmark-port 7778
//
// Packet format (JSON, one per frame):
//   Hand detected:
//     {"command":"ZOOM_IN", "index_x":0.5, "index_y":0.5,
//      "landmarks":[x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]}
//   No hand:
//     {"command":"IDLE", "index_x":0.0, "index_y":0.0, "landmarks":[]}
//
// PUBLIC API (read from any script):
//   Vector3[] JointPositions  — 21 joint positions in HandTracker local space
//   bool      IsTracking      — true while a hand is actively detected
//   string    Command         — "ZOOM_IN" | "ZOOM_OUT" | "PAN" | "IDLE"
//   float     IndexX, IndexY  — MediaPipe normalised [0,1] position of INDEX_TIP

using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace HandGesture
{
    [AddComponentMenu("Hand Gesture/Hand Landmark Receiver")]
    public class HandLandmarkReceiver : MonoBehaviour
    {
        [Header("Network")]
        [Tooltip("UDP port. Must match --landmark-port in Python (default 7778).")]
        [SerializeField] private int listenPort = 7778;

        [Header("Coordinate Mapping")]
        [Tooltip("Scale applied to normalized [0,1] MediaPipe coords → Unity world units.")]
        public float handScale = 2f;
        [Tooltip("Flip X axis if the skeleton appears horizontally mirrored.")]
        public bool flipX = false;

        // ── Public API ──────────────────────────────────────────────────────────
        public const int LandmarkCount = 21;

        /// <summary>
        /// 21 joint positions in the HandTracker GameObject's local space.
        /// Index matches the MediaPipe hand landmark numbering (0 = wrist).
        /// Used by HandSkeletonVisualizer to draw the HUD overlay.
        /// </summary>
        public Vector3[] JointPositions { get; private set; } = new Vector3[LandmarkCount];

        /// <summary>True when Python/MediaPipe has a hand in frame this tick.</summary>
        public bool IsTracking { get; private set; } = false;

        /// <summary>
        /// Classified gesture command from Python's rotation-invariant classifier.
        /// One of: "ZOOM_IN", "ZOOM_OUT", "PAN", "IDLE".
        /// Always "IDLE" when no hand is tracked.
        /// </summary>
        public string Command { get; private set; } = "IDLE";

        /// <summary>
        /// MediaPipe normalised X position of INDEX_TIP (landmark 8), range [0,1].
        /// 0 = left edge of camera image, 1 = right edge.
        /// EarthGestureController consumes this directly for pan deltas.
        /// </summary>
        public float IndexX { get; private set; } = 0.5f;

        /// <summary>
        /// MediaPipe normalised Y position of INDEX_TIP (landmark 8), range [0,1].
        /// 0 = top edge of camera image, 1 = bottom edge.
        /// </summary>
        public float IndexY { get; private set; } = 0.5f;

        // ── Threading ───────────────────────────────────────────────────────────
        private UdpClient _udpClient;
        private Thread _receiveThread;
        private readonly ConcurrentQueue<string> _queue = new ConcurrentQueue<string>();
        private volatile bool _running;

        // ── Lifecycle ───────────────────────────────────────────────────────────
        private void OnEnable()
        {
            _udpClient = new UdpClient(listenPort);
            _running = true;
            _receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "LandmarkReceiverUDP"
            };
            _receiveThread.Start();
            Debug.Log($"[HandLandmarkReceiver] Listening on UDP :{listenPort}");
        }

        private void OnDisable()
        {
            _running = false;
            _udpClient?.Close();
            _receiveThread?.Join(500);
        }

        private void Update()
        {
            // Drain to main thread; keep only the most recent packet per frame
            string latest = null;
            while (_queue.TryDequeue(out string msg))
                latest = msg;

            if (latest != null)
                ParsePacket(latest);
        }

        // ── Private helpers ─────────────────────────────────────────────────────
        private void ReceiveLoop()
        {
            var ep = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    byte[] data = _udpClient.Receive(ref ep);
                    _queue.Enqueue(Encoding.UTF8.GetString(data));
                }
                catch (SocketException) { break; }
                catch (ObjectDisposedException) { break; }
            }
        }

        private void ParsePacket(string json)
        {
            try
            {
                var pkt = JsonUtility.FromJson<LandmarkPacket>(json);

                // Command and index-tip are always carried, even when no hand
                Command = string.IsNullOrEmpty(pkt.command) ? "IDLE" : pkt.command;
                IndexX  = pkt.index_x;
                IndexY  = pkt.index_y;

                // No (or incomplete) landmarks → no tracking
                if (pkt.landmarks == null || pkt.landmarks.Length < LandmarkCount * 3)
                {
                    IsTracking = false;
                    return;
                }

                IsTracking = true;

                for (int i = 0; i < LandmarkCount; i++)
                {
                    float rawX = pkt.landmarks[i * 3];
                    float rawY = pkt.landmarks[i * 3 + 1];
                    float rawZ = pkt.landmarks[i * 3 + 2];

                    // Remap from MediaPipe [0,1] normalized coords to Unity local space:
                    //   X — center at 0.5, flip optional
                    //   Y — flip because MediaPipe Y is top-down, Unity Y is up
                    //   Z — negate so fingers closer to camera have positive Z
                    float mappedX = (flipX ? (1f - rawX) : rawX) - 0.5f;
                    JointPositions[i] = new Vector3(
                        mappedX * handScale,
                        (0.5f - rawY) * handScale,
                        -rawZ * handScale
                    );
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[HandLandmarkReceiver] Parse error: {e.Message}");
            }
        }

        // ── JSON deserialization target  (field names must match JSON keys) ─────
        [Serializable]
        private class LandmarkPacket
        {
            public string  command;     // "ZOOM_IN" | "ZOOM_OUT" | "PAN" | "IDLE"
            public float   index_x;     // [0,1] MediaPipe X of INDEX_TIP
            public float   index_y;     // [0,1] MediaPipe Y of INDEX_TIP
            public float[] landmarks;   // flat: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
        }
    }
}
