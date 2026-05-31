// Assets/Scripts/HandTracking/EarthGestureController.cs ── Version 16
//
// Camera architecture: Main Camera is a child of CesiumGeoreference,
// uses CesiumGlobeAnchor for placement, CesiumOriginShift for precision.
//
// What's new in v16
// ─────────────────
//   • LOCALE-IMMUNE coordinate input.
//     Unity's Inspector parses public double fields using the system culture.
//     On Turkish (and other "decimal comma" locales) "30.5181" silently
//     becomes 30 / 305181 / NaN → camera teleports into the ocean.
//     Fix: store start coords as STRINGS and parse with InvariantCulture
//     (also accepting comma as decimal separator) in Start().
//
//   • Stable pan speed.
//     Linear altitude scaling × tiny MediaPipe jitter × large panSpeed was
//     producing thousand-km-per-frame jumps at high altitudes.
//     Fixes:
//       - Lowered panSpeed default
//       - Per-frame raw delta clamp (caps the worst MediaPipe spike)
//       - Per-frame velocity clamp (capped fraction of altitude per second)
//
//   • DoZoom is UNCHANGED — works perfectly with CesiumOriginShift.

using CesiumForUnity;
using Unity.Mathematics;
using UnityEngine;

namespace HandGesture
{
    [AddComponentMenu("Hand Gesture/Earth Gesture Controller")]
    public class EarthGestureController : MonoBehaviour
    {
        // ════════════════════════════════════════════════════════════════════════
        // Inspector
        // ════════════════════════════════════════════════════════════════════════

        [Header("References")]
        public HandLandmarkReceiver receiver;

        // ── Start position: TURKEY ──────────────────────────────────────────────
        // STRING fields so Unity's locale-dependent Inspector parser can't corrupt
        // the decimal point.  Both '.' and ',' are accepted in the text.
        [Header("Start Position — Locale-Immune (use . OR , for decimals)")]
        [Tooltip("Longitude in degrees east.  Either '.' or ',' works as the decimal separator.\n" +
                 "Example: 30.5181  (or 30,5181 on Turkish keyboards).")]
        public string startLongitudeText = "30.5181";

        [Tooltip("Latitude in degrees north.  Either '.' or ',' works as the decimal separator.\n" +
                 "Example: 39.7711  (or 39,7711 on Turkish keyboards).")]
        public string startLatitudeText = "39.7711";

        [Tooltip("Start altitude in metres above the WGS84 ellipsoid.\n" +
                 "Example: 2500000  or  2500.5")]
        public string startAltitudeText = "2500000";

        [Range(0f, 90f)]
        [Tooltip("Camera pitch on start (degrees below horizontal).\n" +
                 "90° = looking straight down (Nadir) at the Earth.")]
        public float startPitch = 90f;

        [Tooltip("Re-apply the down-look orientation every LateUpdate.\n" +
                 "Defensive lock so nothing can silently rotate the camera.")]
        public bool lockNadirEveryFrame = true;

        // ── Skeleton HUD ─────────────────────────────────────────────────────────
        [Header("Hand Skeleton HUD")]
        public float skeletonDistance = 1.5f;
        public float skeletonOffsetX = 0f;
        public float skeletonOffsetY = -0.15f;

        // ── Zoom ─────────────────────────────────────────────────────────────────
        [Header("Zoom  (anchor.longitudeLatitudeHeight)")]
        [Tooltip("Fraction of altitude per second.\n" +
                 "  0.5 → gentle    1.0 → comfortable    2.0 → aggressive")]
        public float zoomSpeed = 1.0f;
        [Tooltip("Hard floor altitude (street level).")]
        public double minAltitude = 50.0;
        [Tooltip("Hard ceiling altitude — NEVER zoom out past this.\n" +
                 "2 500 000 m keeps Turkey just barely in view.")]
        public double maxAltitude = 2_500_000.0;

        // ── Pan ──────────────────────────────────────────────────────────────────
        [Header("Pan  (clamped to prevent runaway)")]
        [Tooltip("Pan responsiveness.  0.1 = comfortable.  Was 0.5 — that was too hot.")]
        public float panSpeed = 0.1f;
        [Range(1f, 30f)]
        public float panSmoothing = 8f;
        [Tooltip("Hard cap on the per-frame INDEX_TIP viewport delta.\n" +
                 "Filters out MediaPipe jitter spikes that would otherwise " +
                 "translate to thousands-of-km-per-frame jumps.")]
        [Range(0.005f, 0.1f)]
        public float maxDeltaPerFrame = 0.02f;
        [Tooltip("Hard cap on camera movement per second, as a fraction of altitude.\n" +
                 "0.2 → at most 20% of current altitude traversed per second.")]
        [Range(0.05f, 1.0f)]
        public float maxVelocityFractionOfAltitudePerSec = 0.2f;
        public bool invertPanX = false;
        public bool invertPanY = false;

        // ════════════════════════════════════════════════════════════════════════
        // Public state (UI / debug)
        // ════════════════════════════════════════════════════════════════════════

        public string ActiveCommand { get; private set; } = "IDLE";

        // ════════════════════════════════════════════════════════════════════════
        // Private
        // ════════════════════════════════════════════════════════════════════════

        private CesiumGlobeAnchor _anchor;
        private double _targetAltitude;
        private string _prevCommand = "IDLE";

        // The local rotation that points the camera straight down at Earth.
        private Quaternion _lockedLocalRotation;

        // Pan smoothing
        private Vector3 _panVel;
        private float _prevIndexX;
        private float _prevIndexY;
        private bool _panSeeded;

        // ════════════════════════════════════════════════════════════════════════
        // Lifecycle
        // ════════════════════════════════════════════════════════════════════════

        private void Start()
        {
            Application.targetFrameRate = 60;
            if (receiver == null)
                receiver = FindObjectOfType<HandLandmarkReceiver>();
            if (receiver == null)
            {
                Debug.LogError("[EGC] HandLandmarkReceiver not found in scene.");
                enabled = false;
                return;
            }

            // ── Parse start coordinates with InvariantCulture ─────────────────
            // This is the cure for the Turkish-locale teleport bug.  Both
            // "30.5181" and "30,5181" produce the correct double 30.5181.
            double parsedLon = ParseInvariantCulture(startLongitudeText, 30.5181, "startLongitude");
            double parsedLat = ParseInvariantCulture(startLatitudeText, 39.7711, "startLatitude");
            double parsedAlt = ParseInvariantCulture(startAltitudeText, 2_500_000.0, "startAltitude");

            // Sanity-clamp to legal ranges so a typo can never crash Cesium.
            parsedLon = System.Math.Clamp(parsedLon, -180.0, 180.0);
            parsedLat = System.Math.Clamp(parsedLat,  -85.0,  85.0);
            parsedAlt = System.Math.Clamp(parsedAlt, minAltitude, maxAltitude);

            // ── Get / add the globe anchor ─────────────────────────────────────
            _anchor = GetComponent<CesiumGlobeAnchor>();
            if (_anchor == null)
                _anchor = gameObject.AddComponent<CesiumGlobeAnchor>();

            // ── CRITICAL: configure anchor BEFORE the teleport ─────────────────
            // adjustOrientationForGlobeWhenMoving defaults to TRUE in Cesium,
            // which makes the anchor silently rotate the camera every time
            // longitudeLatitudeHeight is written.  Disabling it before the
            // teleport keeps our orientation sacred.
            _anchor.adjustOrientationForGlobeWhenMoving = false;
            _anchor.detectTransformChanges = true;

            // ── TELEPORT (atomic double3 write) ────────────────────────────────
            _anchor.longitudeLatitudeHeight = new double3(parsedLon, parsedLat, parsedAlt);
            _targetAltitude = parsedAlt;

            // ── Nadir orientation lock ─────────────────────────────────────────
            InitializeCameraOrientation();

            Debug.Log($"[EGC v16] Start — lon={parsedLon:F4}° lat={parsedLat:F4}° " +
                      $"alt={parsedAlt:F0} m  pos={transform.position}  " +
                      $"localRot={transform.localEulerAngles}");
        }

        /// <summary>
        /// Robust, locale-independent double parser.
        ///
        ///   • Normalises ',' → '.' so users in comma-decimal locales (Turkish,
        ///     German, French, …) can type either style.
        ///   • Parses with System.Globalization.CultureInfo.InvariantCulture
        ///     so the system's culture cannot reinterpret the value.
        ///   • Falls back to a sane default on parse failure (with a warning).
        /// </summary>
        private static double ParseInvariantCulture(string text, double fallback, string fieldName)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                Debug.LogWarning($"[EGC] {fieldName} is empty; using fallback {fallback}.");
                return fallback;
            }

            string normalised = text.Trim().Replace(',', '.');
            if (double.TryParse(
                    normalised,
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out double value))
            {
                return value;
            }

            Debug.LogWarning($"[EGC] Could not parse {fieldName}='{text}'; " +
                             $"using fallback {fallback}.");
            return fallback;
        }

        private void InitializeCameraOrientation()
        {
            _lockedLocalRotation = Quaternion.Euler(startPitch, 0f, 0f);
            transform.localRotation = _lockedLocalRotation;
        }

        // ════════════════════════════════════════════════════════════════════════
        // Update — command dispatch
        // ════════════════════════════════════════════════════════════════════════

        private void Update()
        {
            string cmd = receiver != null ? receiver.Command : "IDLE";
            ActiveCommand = cmd;

            Debug.Log($"[Gesture] Cmd: {cmd} | Target: {_targetAltitude:F0} | " +
                      $"CamY: {transform.position.y:F0}");

            if (cmd != _prevCommand)
            {
                _panSeeded = false;
                _panVel = Vector3.zero;
                _prevCommand = cmd;
            }

            switch (cmd)
            {
                case "ZOOM_IN":
                    _panVel = Vector3.zero; _panSeeded = false;
                    DoZoom(zoomIn: true);
                    break;

                case "ZOOM_OUT":
                    _panVel = Vector3.zero; _panSeeded = false;
                    DoZoom(zoomIn: false);
                    break;

                case "PAN":
                    DoPan();
                    break;

                default:   // IDLE / unknown
                    BleedPanInertia();
                    break;
            }
        }

        private void LateUpdate()
        {
            if (lockNadirEveryFrame)
                transform.localRotation = _lockedLocalRotation;

            if (receiver == null) return;
            Transform ht = receiver.transform;

            // KESİN ÇÖZÜM: El modelini Kameranın içine (child objesi olarak) kilitliyoruz.
            // Böylece Cesium dünyayı veya kamerayı ışınlasa bile, el modeli asla arkada kalmaz.
            if (ht.parent != transform)
            {
                ht.SetParent(transform, true);
            }

            // Artık Dünya koordinatları (position) yerine, doğrudan Kameraya olan uzaklığı (localPosition) kullanıyoruz.
            ht.localPosition = new Vector3(skeletonOffsetX, skeletonOffsetY, skeletonDistance);
            ht.localRotation = Quaternion.identity;
        }

        // ════════════════════════════════════════════════════════════════════════
        // Zoom — UNCHANGED (anchor.longitudeLatitudeHeight, Cesium-native)
        // ════════════════════════════════════════════════════════════════════════

        private void DoZoom(bool zoomIn)
        {
            // 1. Cesium'dan kameranın Dünya üzerindeki gerçek coğrafi yüksekliğini al
            double3 llh = _anchor.longitudeLatitudeHeight;

            // 2. Matematiksel sıfır kilitlenmesini engelle
            double baseAlt = System.Math.Max(100.0, llh.z);
            double step = baseAlt * (double)zoomSpeed * (double)Time.deltaTime;

            // 3. Eli açıp kapatmaya göre hedef yüksekliği belirle
            double newAlt = llh.z;
            if (zoomIn) newAlt -= step;
            else newAlt += step;

            // 4. Uzay boşluğuna fırlamayı engellemek için sınırla
            newAlt = System.Math.Clamp(newAlt, minAltitude, maxAltitude);

            // 5. ÇALIŞAN KISIM: Kameranın Y değerini DEĞİL, Dünya'nın yüksekliğini değiştiriyoruz
            _anchor.longitudeLatitudeHeight = new double3(llh.x, llh.y, newAlt);

            // 6. Logların doğru görünmesi için eşitle
            _targetAltitude = newAlt;
        }

        // ════════════════════════════════════════════════════════════════════════
        // Pan — clamped to prevent runaway velocity
        // ════════════════════════════════════════════════════════════════════════
        //
        // Three-stage safety:
        //   1) Raw INDEX_TIP delta clamped per frame (kills MediaPipe spikes)
        //   2) Velocity computed in altitude-proportional terms
        //   3) Velocity capped at fraction of altitude per second (hard ceiling)

        private void DoPan()
        {
            float cx = receiver.IndexX;
            float cy = receiver.IndexY;

            if (!_panSeeded)
            {
                _prevIndexX = cx;
                _prevIndexY = cy;
                _panSeeded = true;
                return;
            }

            float rawDx = cx - _prevIndexX;
            float rawDy = cy - _prevIndexY;
            _prevIndexX = cx;
            _prevIndexY = cy;

            // ── ① Clamp the raw per-frame delta ──────────────────────────────
            // MediaPipe occasionally spikes the index-tip position by 0.1–0.5
            // in a single frame.  Without this clamp, those spikes become
            // hundreds-of-km jumps at high altitude.
            rawDx = Mathf.Clamp(rawDx, -maxDeltaPerFrame, maxDeltaPerFrame);
            rawDy = Mathf.Clamp(rawDy, -maxDeltaPerFrame, maxDeltaPerFrame);

            float dx =  rawDx * (invertPanX ? -1f :  1f);
            float dy = -rawDy * (invertPanY ? -1f :  1f);   // MediaPipe Y+ is down

            // ── ② True geographic altitude (CesiumOriginShift-safe) ──────────
            double trueAltitude = _anchor.longitudeLatitudeHeight.z;
            float alt = Mathf.Max(100f, (float)trueAltitude);

            // Horizontal basis built from world-up × camera-right (cross product
            // works regardless of pitch; transform.forward is zero when flattened
            // because we're staring straight down at the ground).
            Vector3 right = transform.right; right.y = 0f;
            if (right.sqrMagnitude > 0.001f) right.Normalize();
            Vector3 forward = Vector3.Cross(right, Vector3.up).normalized;

            Vector3 target = (right * dx + forward * dy) * panSpeed * alt;

            // ── ③ Hard cap on velocity magnitude ─────────────────────────────
            // Even with clamped delta, very large altitudes × many frames can
            // produce uncomfortable jumps.  Cap absolute velocity at a fraction
            // of altitude per second.
            float maxStep = alt * maxVelocityFractionOfAltitudePerSec * Time.deltaTime;
            if (target.magnitude > maxStep)
                target = target.normalized * maxStep;

            _panVel = Vector3.Lerp(_panVel, target, panSmoothing * Time.deltaTime);
            transform.position += _panVel;
        }

        private void BleedPanInertia()
        {
            _panVel = Vector3.Lerp(_panVel, Vector3.zero, panSmoothing * Time.deltaTime);
            if (_panVel.sqrMagnitude > 0.0001f)
                transform.position += _panVel;
        }
    }
}
