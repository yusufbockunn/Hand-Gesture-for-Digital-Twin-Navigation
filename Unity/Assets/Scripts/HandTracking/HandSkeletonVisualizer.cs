// Assets/Scripts/HandTracking/HandSkeletonVisualizer.cs
// Renders the hand skeleton: 21 sphere joints + 21 color-coded bone LineRenderers.
// Reads joint positions from HandLandmarkReceiver every frame.
//
// PHASE-2 NOTES:
//   Each joint is a named child GameObject: "Joint_00" through "Joint_20".
//   SphereColliders are created but disabled — enable them per-joint in your
//   interaction script using GetJointObject(i).GetComponent<SphereCollider>().
//   Use GetJointWorldPosition(i) for raycasts / pinch-distance calculations.

using UnityEngine;
using UnityEngine.Rendering;

namespace HandGesture
{
    [RequireComponent(typeof(HandLandmarkReceiver))]
    [AddComponentMenu("Hand Gesture/Hand Skeleton Visualizer")]
    public class HandSkeletonVisualizer : MonoBehaviour
    {
        [Header("Joint Appearance")]
        [Tooltip("Radius of each joint sphere (Unity units).")]
        public float jointRadius = 0.04f;
        [Tooltip("Optional shared material for all joint spheres. Leave empty for auto.")]
        public Material jointMaterial;

        [Header("Bone Appearance")]
        [Tooltip("Width of each bone line (Unity units).")]
        public float boneWidth = 0.018f;
        [Tooltip("Optional base material for bone lines. Leave empty for auto.")]
        public Material boneMaterial;

        [Header("Behavior")]
        [Tooltip("Hide the skeleton when Python reports no hand in frame.")]
        public bool hideWhenNotTracking = true;

        // ── MediaPipe hand topology — 21 bone connections ───────────────────────
        // Indices match the standard MediaPipe 21-landmark numbering.
        private static readonly int[,] Bones =
        {
            // Palm ring (connects wrist → knuckles)
            { 0,  1}, { 1,  2}, { 2,  5}, { 5,  9}, { 9, 13}, {13, 17}, {17,  0},
            // Thumb  (CMC → MCP → IP → TIP)
            { 2,  3}, { 3,  4},
            // Index  (MCP → PIP → DIP → TIP)
            { 5,  6}, { 6,  7}, { 7,  8},
            // Middle
            { 9, 10}, {10, 11}, {11, 12},
            // Ring
            {13, 14}, {14, 15}, {15, 16},
            // Pinky
            {17, 18}, {18, 19}, {19, 20}
        };

        // ── Per-finger color coding ─────────────────────────────────────────────
        private static readonly Color[] FingerColors =
        {
            new Color(0.85f, 0.85f, 0.85f, 1f), // 0: palm   — light grey
            new Color(1.00f, 0.45f, 0.45f, 1f), // 1: thumb  — red
            new Color(0.45f, 1.00f, 0.45f, 1f), // 2: index  — green
            new Color(0.45f, 0.65f, 1.00f, 1f), // 3: middle — blue
            new Color(1.00f, 1.00f, 0.45f, 1f), // 4: ring   — yellow
            new Color(0.45f, 1.00f, 1.00f, 1f), // 5: pinky  — cyan
        };

        // Maps bone index → FingerColors index (must match Bones array order above)
        private static readonly int[] BoneColorGroup =
            { 0, 0, 0, 0, 0, 0, 0,   // 7 palm bones
              1, 1,                    // 2 thumb bones
              2, 2, 2,                 // 3 index bones
              3, 3, 3,                 // 3 middle bones
              4, 4, 4,                 // 3 ring bones
              5, 5, 5 };               // 3 pinky bones

        // ── Runtime references ──────────────────────────────────────────────────
        private HandLandmarkReceiver _receiver;
        private GameObject[]   _joints;
        private LineRenderer[] _bones;

        // ── Lifecycle ───────────────────────────────────────────────────────────
        private void Awake()
        {
            _receiver = GetComponent<HandLandmarkReceiver>();
            BuildSkeleton();
        }

        private void Update()
        {
            bool tracking = _receiver.IsTracking;

            if (hideWhenNotTracking)
                SetVisible(tracking);

            if (!tracking) return;

            var pos = _receiver.JointPositions;
            int boneCount = Bones.GetLength(0);

            for (int i = 0; i < HandLandmarkReceiver.LandmarkCount; i++)
                _joints[i].transform.localPosition = pos[i];

            for (int b = 0; b < boneCount; b++)
            {
                _bones[b].SetPosition(0, pos[Bones[b, 0]]);
                _bones[b].SetPosition(1, pos[Bones[b, 1]]);
            }
        }

        // ── Public API for Phase 2 ──────────────────────────────────────────────

        /// <summary>
        /// World-space position of joint <paramref name="index"/> (0–20).
        /// Use for pinch-distance checks, raycasts, and interaction logic.
        /// </summary>
        public Vector3 GetJointWorldPosition(int index)
            => _joints[index].transform.position;

        /// <summary>
        /// The child GameObject for joint <paramref name="index"/> (0–20).
        /// Enable its SphereCollider here to participate in physics / triggers.
        /// </summary>
        public GameObject GetJointObject(int index)
            => _joints[index];

        // ── Private helpers ─────────────────────────────────────────────────────
        private void BuildSkeleton()
        {
            int count = HandLandmarkReceiver.LandmarkCount;
            _joints = new GameObject[count];

            Material jMat = jointMaterial != null
                ? jointMaterial
                : MakeUnlitMaterial(Color.white);

            // Joint spheres
            for (int i = 0; i < count; i++)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = $"Joint_{i:D2}";
                go.transform.SetParent(transform, worldPositionStays: false);
                go.transform.localScale = Vector3.one * (jointRadius * 2f);

                // Disable collider — interaction scripts re-enable per joint in Phase 2
                var col = go.GetComponent<SphereCollider>();
                if (col != null) col.enabled = false;

                go.GetComponent<Renderer>().material = jMat;
                _joints[i] = go;
            }

            // Bone LineRenderers
            int boneCount = Bones.GetLength(0);
            _bones = new LineRenderer[boneCount];

            for (int b = 0; b < boneCount; b++)
            {
                var go = new GameObject($"Bone_{b:D2}");
                go.transform.SetParent(transform, worldPositionStays: false);

                var lr = go.AddComponent<LineRenderer>();
                lr.positionCount = 2;
                lr.startWidth = lr.endWidth = boneWidth;
                lr.useWorldSpace = false; // positions are in HandTracker local space
                lr.shadowCastingMode = ShadowCastingMode.Off;
                lr.receiveShadows = false;

                Color col = FingerColors[BoneColorGroup[b]];
                Material mat = boneMaterial != null
                    ? new Material(boneMaterial)
                    : MakeUnlitMaterial(col);
                mat.color = col;
                lr.material = mat;

                _bones[b] = lr;
            }
        }

        private void SetVisible(bool visible)
        {
            if (_joints == null) return;
            foreach (var j in _joints) j.SetActive(visible);
            foreach (var b in _bones)  b.enabled = visible;
        }

        private static Material MakeUnlitMaterial(Color color)
        {
            // Try URP first, fall back to legacy built-in shaders
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit")
                         ?? Shader.Find("Unlit/Color")
                         ?? Shader.Find("Standard");
            var mat = new Material(shader);
            mat.color = color;
            return mat;
        }
    }
}
