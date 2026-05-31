// GestureReceiver.cs
// Listens for gesture command payloads sent by the Python app (utils/unity_bridge.py)
// over UDP and exposes them to the rest of the Unity scene.
//
// Setup
// -----
//   1. Attach this component to any persistent GameObject (e.g. GameManager).
//   2. Set Port to match --unity-port in the Python launch command (default 7777).
//   3. Wire OnGestureReceived or the individual command events in the Inspector,
//      OR read CurrentPayload / CurrentCommand from other scripts every frame.
//
// Python launch example:
//   python app.py --control-unity --unity-port 7777
//
// Incoming JSON format (matches command_payload() in utils/gesture_command.py):
//   {
//     "command":        "PAN_UP",
//     "confidence":     0.85,
//     "source":         "open_hand_move",
//     "hand_sign":      "Open",
//     "finger_gesture": "Move",
//     "earth_action":   "Pan north",
//     "ux_state":       "NAVIGATING"
//   }

using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Events;

namespace HandGesture
{
    // ── Payload struct (must match Python JSON field names exactly) ──────────────

    [Serializable]
    public class GesturePayload
    {
        public string command        = "NONE";
        public float  confidence     = 0f;
        public string source         = "";
        public string hand_sign      = "";
        public string finger_gesture = "";
        public string earth_action   = "";
        public string ux_state       = "IDLE";
    }

    // ── Commands (mirrors gesture_command.py constants) ──────────────────────────

    public static class GestureCommand
    {
        public const string None          = "NONE";
        public const string ZoomIn        = "ZOOM_IN";
        public const string ZoomOut       = "ZOOM_OUT";
        public const string PanUp         = "PAN_UP";
        public const string PanDown       = "PAN_DOWN";
        public const string PanLeft       = "PAN_LEFT";
        public const string PanRight      = "PAN_RIGHT";
        public const string RotateLeft    = "ROTATE_LEFT";
        public const string RotateRight   = "ROTATE_RIGHT";
        public const string ResetView     = "RESET_VIEW";
        public const string Select        = "SELECT";
        public const string Stop          = "STOP";
        public const string Close         = "CLOSE";
    }

    // ── Component ────────────────────────────────────────────────────────────────

    [AddComponentMenu("Hand Gesture/Gesture Receiver")]
    public class GestureReceiver : MonoBehaviour
    {
        [Header("Network")]
        [Tooltip("UDP port to listen on. Must match --unity-port in the Python launch command.")]
        [SerializeField] private int port = 7777;

        [Tooltip("Start listening automatically when the component is enabled.")]
        [SerializeField] private bool startOnEnable = true;

        // ── Events ───────────────────────────────────────────────────────────────

        [Header("Events")]
        [Tooltip("Fired every frame a payload arrives (including NONE / idle frames).")]
        public UnityEvent<GesturePayload> onGestureReceived;

        [Space(4)]
        [Tooltip("Fired once when any active command starts (command != NONE).")]
        public UnityEvent<string> onCommandStarted;

        [Tooltip("Fired once when the command returns to NONE.")]
        public UnityEvent onCommandStopped;

        // ── Public state (poll from other scripts) ───────────────────────────────

        /// <summary>Most recent payload from Python. Updated on the main thread.</summary>
        public GesturePayload CurrentPayload { get; private set; } = new GesturePayload();

        /// <summary>Shortcut for CurrentPayload.command.</summary>
        public string CurrentCommand => CurrentPayload?.command ?? GestureCommand.None;

        /// <summary>True while the UDP listener thread is running.</summary>
        public bool IsRunning { get; private set; }

        // ── Private ──────────────────────────────────────────────────────────────

        private UdpClient  _udpClient;
        private Thread     _thread;
        private CancellationTokenSource _cts;
        private readonly ConcurrentQueue<string> _queue = new ConcurrentQueue<string>();
        private string _lastCommand = GestureCommand.None;

        // ── Lifecycle ────────────────────────────────────────────────────────────

        private void OnEnable()
        {
            if (startOnEnable) StartListening();
        }

        private void OnDisable()
        {
            StopListening();
        }

        // ── Public API ───────────────────────────────────────────────────────────

        public void StartListening()
        {
            if (IsRunning) return;

            _cts       = new CancellationTokenSource();
            _udpClient = new UdpClient(port);
            _thread    = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name         = "GestureReceiverUDP"
            };
            _thread.Start();
            IsRunning = true;
            Debug.Log($"[GestureReceiver] Listening on UDP port {port}.");
        }

        public void StopListening()
        {
            if (!IsRunning) return;

            _cts?.Cancel();
            _udpClient?.Close();
            _thread?.Join(500);
            IsRunning = false;
            Debug.Log("[GestureReceiver] Stopped.");
        }

        // ── Main-thread dispatch (Update) ────────────────────────────────────────

        private void Update()
        {
            // Drain all queued JSON strings (network thread → main thread)
            while (_queue.TryDequeue(out string json))
            {
                GesturePayload payload;
                try
                {
                    payload = JsonUtility.FromJson<GesturePayload>(json);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[GestureReceiver] JSON parse error: {e.Message}  raw={json}");
                    continue;
                }

                CurrentPayload = payload;
                onGestureReceived?.Invoke(payload);

                // Edge-detect command changes to fire started / stopped events
                if (payload.command != _lastCommand)
                {
                    if (payload.command == GestureCommand.None)
                        onCommandStopped?.Invoke();
                    else
                        onCommandStarted?.Invoke(payload.command);

                    _lastCommand = payload.command;
                }
            }
        }

        // ── Background receive loop ──────────────────────────────────────────────

        private void ReceiveLoop()
        {
            var endpoint = new IPEndPoint(IPAddress.Any, 0);
            while (!_cts.IsCancellationRequested)
            {
                try
                {
                    byte[] bytes = _udpClient.Receive(ref endpoint);
                    _queue.Enqueue(Encoding.UTF8.GetString(bytes));
                }
                catch (SocketException)
                {
                    break; // socket closed — normal shutdown path
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
            }
        }
    }
}
