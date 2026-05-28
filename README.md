# 🌍 Hand Gesture Controlled Earth Navigation System

<div align="center">

![Unity](https://img.shields.io/badge/Unity-2022+-black?style=for-the-badge\&logo=unity)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-orange?style=for-the-badge)
![Cesium](https://img.shields.io/badge/Cesium-3D%20GIS-green?style=for-the-badge)
![UDP](https://img.shields.io/badge/Protocol-UDP-red?style=for-the-badge)

### Real-Time Gesture-Based Geospatial Navigation using Computer Vision & Cesium for Unity

Navigate the Earth using only your hands — no mouse, no controller, no VR hardware.

Inspired by **Minority Report**, **Google Earth**, and futuristic Human-Computer Interaction systems.

</div>

---

# 📺 Demo & Preview

## 🌆 Project Screenshot

 <img width="1915" height="861" alt="image" src="https://github.com/user-attachments/assets/672b7a42-79ed-457f-a7dd-a03ad5108e18" /> <img width="1919" height="938" alt="image" src="https://github.com/user-attachments/assets/7307a211-1c55-4e2f-94a2-d1dfacf22d39" />



---

## 🎥 Demo Video

▶️ Youtube Link

```md
[Watch Demo Video]
```

---

# 🧠 Project Overview

This project is a **real-time geographical simulation and interaction system** that enables users to navigate a full-scale 3D Earth using standard webcam hand gestures.

The system combines:

* 🖐️ **Computer Vision Hand Tracking**
* 🌍 **Real-Time Geospatial Rendering**
* 📡 **Low-Latency UDP Networking**
* 🧭 **Adaptive Camera Navigation**
* 🛰️ **Cesium's WGS84 Globe System**

A Python backend powered by **MediaPipe** detects and classifies hand gestures, while a Unity frontend powered by **Cesium for Unity** visualizes and navigates a photorealistic Earth environment.

---

# 🏗️ System Architecture

```text
[ Webcam Stream ]
       │
       ▼
[ Python Backend (MediaPipe) ]
       │
       ├─► 21 Joint Coordinate Extraction
       ├─► Gesture Classification Engine
       └─► UDP JSON Streaming (Port 7778)
                        │
                        ▼
[ Unity Frontend (Cesium for Unity) ]
       │
       ├─► Thread-safe Queue Receiver
       ├─► Geographic Coordinate Transform
       ├─► Adaptive Camera Controller
       └─► Real-Time Earth Navigation
```

---

# ⚙️ Technology Stack

| Layer             | Technology                       |
| ----------------- | -------------------------------- |
| Computer Vision   | MediaPipe Hands                  |
| Backend           | Python                           |
| Networking        | UDP Sockets                      |
| Frontend          | Unity                            |
| GIS Engine        | Cesium for Unity                 |
| 3D Globe          | Google Photorealistic 3D Tiles   |
| Coordinate System | WGS84 Ellipsoid                  |
| Concurrency       | ConcurrentQueue / Multithreading |

---

# 🖐️ Python Backend

The Python backend acts as the **gesture processing and data provider layer**.

## 🔍 Features

### ✅ Real-Time Hand Tracking

Using **MediaPipe Hands**, the system extracts:

* 21 hand landmarks
* 3D joint positions
* fingertip coordinates
* gesture states

all in real time with extremely low latency.

---

### 🧠 Gesture Classification Engine

The classifier analyzes:

* Relative landmark distances
* Finger states
* Joint angles
* Motion patterns

to generate four core commands:

| Gesture | Action     |
| ------- | ---------- |
| ✊       | `ZOOM_IN`  |
| 🖐️     | `ZOOM_OUT` |
| ☝️      | `PAN`      |
| Idle    | `IDLE`     |

---

### 📡 UDP Streaming

Processed gesture data is serialized into JSON packets and streamed to Unity at ~30 FPS.

Example packet:

```json
{
  "command": "PAN",
  "index_x": 0.52,
  "index_y": 0.31
}
```

---

# 🌍 Unity Frontend

The Unity application acts as the **visualization and simulation core**.

Powered by **Cesium for Unity**, the frontend renders the entire Earth with photorealistic 3D streaming.

---

## 🌐 Cesium Integration

### Features

* Real-scale globe rendering
* WGS84 geospatial coordinates
* Google Photorealistic 3D Tiles
* Infinite world streaming
* Precision-safe global traversal

---

## 📉 Floating Origin Problem Solution

Large-scale worlds suffer from floating-point precision loss.

This project solves that using:

* `CesiumGlobeAnchor`
* `CesiumOriginShift`

The world origin is dynamically re-centered to maintain:

* Rendering stability
* Physics stability
* Accurate transformations
* Jitter-free movement

even when traveling across continents.

---

# 🚀 Key Technical Features

## 🌐 Locale-Independent Coordinate Parsing

Operating systems using non-US locales (Turkish, German, French, etc.) may interpret decimal separators incorrectly.

Example:

```text
30.5181 → 305181
```

This causes catastrophic coordinate corruption.

### ✅ Solution

Coordinates are parsed using:

```csharp
CultureInfo.InvariantCulture
```

while normalizing both:

* `,`
* `.`

decimal separators safely.

---

# 📉 Adaptive Damping & Differential Pan System

The system dynamically scales movement speed according to the camera’s real geographic altitude.

## High Altitude

At:

```text
2,500,000m
```

small hand movement results in continental-scale transitions.

---

## Low Altitude

At:

```text
1,500m
```

the same movement enables precise street-level navigation.

---

## ✨ Jitter Protection Pipeline

Movement smoothing uses a 3-stage defense mechanism:

1. Raw Delta Clamp
2. Linear Altitude Scaling
3. Velocity Ceiling

This creates stable and cinematic motion.

---

# 📐 Nadir View Orientation Lock

When looking directly downward:

```text
Pitch ≈ 90°
```

traditional forward vectors collapse due to gimbal-related directional degeneracy.

### ✅ Solution

The system computes movement using:

```csharp
Vector3.Cross(transform.right, worldUp)
```

which guarantees stable horizontal panning regardless of camera pitch.

---

# 🦴 Persistent Skeleton HUD

When Cesium performs an Origin Shift, world-space overlays normally jitter or disappear.

### ✅ Solution

The hand skeleton renderer is:

* Parent-attached to the Main Camera
* Rendered using `localPosition`
* Fully decoupled from shifting world coordinates

Result:

✔ Stable overlay
✔ No jitter
✔ Persistent HUD visibility

---

# 🛠️ Installation

# 1️⃣ Python Backend Setup

Install dependencies:

```bash
pip install opencv-python mediapipe
```

Run the backend:

```bash
python app.py --stream-landmarks --landmark-port 7778
```

---

# 2️⃣ Unity Frontend Setup

## Clone Repository

```bash
git clone https://github.com/your-username/earth-gesture-navigation.git
```

---

## Open Unity Project

Add the `Unity/` folder into Unity Hub.

---

## Configure Scene

Ensure the following objects are active:

* `CesiumGeoreference`
* `Google Photorealistic 3D Tiles`

---

## Configure Start Location

Example (Central Eskişehir):

| Field     | Value     |
| --------- | --------- |
| Longitude | `30.5181` |
| Latitude  | `39.7711` |
| Altitude  | `2000`    |

---

## Run

Press:

```text
▶ Play
```

inside Unity Editor.

---

# 📁 Repository Structure

```text
├── .git
├── .gitignore
├── README.md
├── hand-gesture-backend/
│   ├── app.py
│   ├── classifier.py
│   └── networking/
│
└── Unity/
    └── Assets/
        └── Scripts/
            └── HandTracking/
                ├── HandLandmarkReceiver.cs
                ├── HandSkeletonVisualizer.cs
                └── EarthGestureController.cs
```

---

# 🎯 Future Improvements

* ✋ Multi-hand gesture support
* 🥽 VR integration
* 🤖 AI gesture prediction
* 🌎 Multiplayer collaborative navigation
* 🧭 Gesture customization UI
* 📱 Mobile camera streaming

---

# 🧪 Research & Academic Context

This project was developed as an exploration of:

* Human-Computer Interaction (HCI)
* Computer Vision
* Real-Time GIS Systems
* Spatial Computing
* Gesture-Based Interfaces
* Large-Scale Simulation Architectures

It demonstrates how modern CV pipelines can seamlessly integrate with advanced geospatial engines for immersive interaction systems.

# ⭐ Acknowledgements

* [MediaPipe](https://github.com/google/mediapipe)
* [Cesium for Unity](https://cesium.com/platform/cesium-for-unity/)
* [Google Photorealistic 3D Tiles](https://developers.google.com/maps/documentation/tile)
* Unity Technologies

---

# 📜 License

This project is licensed under the MIT License.

```text
MIT License © 2026 Yusuf Böçkün-Cem Levent Avcı
```

---
