# Theft Sentinel — Real-Time AI-Powered Retail Theft Detection & Multi-Camera Tracking

> **Final Year Project (FYP)** — A production-grade, real-time retail security monitoring system that ingests live RTSP camera streams, runs a multi-model computer-vision pipeline (YOLOv8 → DeepSORT → OSNet-AIN → FAISS → X3D-S) to detect theft behavior, and surfaces alerts to security personnel through a role-stratified web dashboard with live SSE-based bounding-box overlays.

---

## 1. Research Problem & Core Question

### Research Problem

Traditional retail surveillance relies on passive, post-event CCTV review, which is labor-intensive and reactive. Existing automated systems either (a) perform single-camera person detection without cross-camera identity persistence, or (b) run heavy action-recognition models that cannot maintain a persistent suspect identity across multiple camera views and suffer from false positives due to short clip windows. There is no integrated system that (i) persistently tracks a suspect's identity across multiple cameras under lighting/domain shifts, (ii) recognizes theft behavior from a sufficiently long temporal clip, and (iii) delivers real-time, low-latency visual alerts to guards without flooding the database.

### Core Research Question

> *Can a fused multi-camera Re-ID pipeline (OSNet-AIN + FAISS) combined with a per-identity, 64-frame X3D-S behavioral state machine deliver real-time, cross-camera theft detection with a false-positive rate low enough for live deployment, while remaining memory-safe on a 6 GB GPU?*

### How This Repository Resolves It

The system answers **yes** by implementing four mechanisms, all visible in the codebase:

1. **Persistent cross-camera identity** — `osnet_ain_x1_0` ReID embeddings are mean-centered before L2 normalization and matched via a dual FAISS + identity-DB strategy (`theft_sentinel_backend/ai_pipeline/matching/matcher.py`, `database/identity_db.py`). FAISS embeddings are **never deleted** when a DeepSORT track drops, so a suspect re-entering any camera is re-recognized under the same `global_id`.
2. **Robust per-identity behavioral classification** — `GlobalTheftDetector` (`ai_pipeline/theft_detection/x3d_detector.py`) maintains one `TheftState` per `global_id` with a 64-frame RGB buffer, smoothed score (5-frame rolling mean), consecutive-confirmation gate (3 hits), and an 8 s cooldown — exactly matching the training preprocessing in `step1_preprocess_yolo_crop.py`.
3. **Real-time, throttled delivery** — SSE pub/sub (`sse_registry.py`) pushes tracking payloads at 10 FPS to `CameraFeedWithOverlay.jsx`, which draws LERP-smoothed RED boxes with a 3.5 s TTL grace period, eliminating flicker.
4. **Memory safety on constrained hardware** — immediate 640×480 downscale, `gc.collect()` before/after clip encoding, `torch.cuda.empty_cache()` every 15 frames and before X3D, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see `AI_ENGINE_INTEGRATION.md` §4).

---

## 2. Project Overview & Significance

**Theft Sentinel** is a full-stack, role-stratified retail security platform. A Django REST backend (MongoDB via `django-mongodb-backend`) exposes JWT-secured APIs for cameras, alerts, incidents, personnel, and AI control. A React 18 + Vite SPA provides dashboards and a Control Room with live MJPEG feeds overlaid with AI bounding boxes. The AI engine runs in background daemon threads (decoupled from the request cycle) per camera, persisting alert clips to Cloudinary.

### Significance & Novelty

- **Cross-camera identity persistence** via FAISS permanent memory + relaxed DeepSORT distance thresholds — suspects are not forgotten when they leave frame.
- **Training/deploy parity** — X3D preprocessing (`_frames_to_tensor`) is bit-identical to the training pipeline (64 frames, 224×224, Kinetics-400 normalization), avoiding the train/deploy mismatch that plagues most deployed CV systems.
- **Identity-hijacking prevention** — an IoU-based `used_global_ids` pre-pass rejects assigning an already-in-frame `global_id` to a second person.
- **Dual-synchronization "Stop Tracking"** — backend registry removal + frontend `CustomEvent` wipe clears a suspect across all camera canvases instantly.
- **Multi-tenancy & RBAC** — `ADMIN`, `SECURITY_INCHARGE`, `SECURITY_GUARD` roles with branch-level authorization enforced on AI monitor start/stop.

---

## 3. Key Features

- **Live RTSP/HTTP camera ingestion** with auto protocol detection and exponential-backoff reconnect (`continuous_monitor.py`, `stream_manager`).
- **YOLOv8 person detection** at 640×480 with configurable confidence threshold.
- **DeepSORT per-camera tracking** with isolated tracker state per `InferenceRunner` instance.
- **OSNet-AIN Re-ID** with mean-centering + 10-frame embedding smoothing for cross-camera matching.
- **FAISS + identity-DB dual matching** with same-cam (0.60) and diff-cam (0.50) thresholds.
- **X3D-S theft classification** per global identity with 64-frame buffer, smoothed scoring, consecutive confirmation, and 8 s cooldown.
- **Real-time SSE streaming** of tracking payloads (10 FPS) with keepalive comments every 25 s.
- **Live MJPEG feed** with canvas overlay, LERP animation, 3.5 s TTL grace period, NaN guards.
- **Alert clip recording** — rolling 150-frame buffer snapshot, MP4 encode via OpenCV, Cloudinary upload in a daemon thread.
- **Alert cooldown gatekeeper** (5 s) preventing DB flooding at camera FPS.
- **Role-based dashboards** — Overview, Historical Reporting, Guard Dashboard, Control Room, Alerts, Incidents, AI Dashboard.
- **Incident management** — assignment, status transitions (CREATED → ASSIGNED → ACKNOWLEDGED → RESOLVED), notes.
- **Mobile notifications** — SMS (Twilio) and email alerts for admins/incharges.
- **Feedback system** — guard-submitted false/true positive reports feeding model evaluation.
- **Multi-tenancy** — branch-scoped camera access and per-tenant data isolation.
- **AI Health & Model Info endpoints** — load status of YOLO, OSNet, X3D, device info.

---

## 4. System Architecture & Tech Stack

### Architecture

**Client-Server with a decoupled AI pipeline.** The Django REST backend owns persistence, auth, and camera metadata. The React SPA handles presentation. The AI engine runs in background daemon threads managed by a process-level `MonitorManager` singleton, publishing to SSE clients via a thread-safe `SSERegistry`. MongoDB is the primary data store; Cloudinary hosts alert clips.

```
RTSP Cameras → ContinuousMonitor (capture + inference threads)
            → YOLO → DeepSORT → OSNet → FAISS → X3D
            → SSE Registry → React Canvas Overlay
            → Alert/Incident → MongoDB + Cloudinary
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Django 5.2 + Django REST Framework 3.16 |
| **Database** | MongoDB Atlas (`django-mongodb-backend 5.2`) |
| **Auth** | JWT (SimpleJWT) — access/refresh tokens in localStorage |
| **AI / CV** | PyTorch 2.6 (CUDA 12.4), OpenCV 4.9, Ultralytics YOLOv8, DeepSORT-Realtime, TorchReID (OSNet-AIN), PyTorchVideo (X3D), FAISS-CPU |
| **Video Storage** | Cloudinary (alert clip hosting) |
| **Notifications** | Twilio (SMS), Django SMTP (email) |
| **Frontend** | React 18.2 + Vite 8 |
| **State Management** | Recoil |
| **Styling** | Tailwind CSS 3.4 + custom design tokens |
| **Charts** | Recharts |
| **Real-Time Push** | Server-Sent Events (Django `StreamingHttpResponse`) |
| **Mobile (optional)** | Capacitor (Android) |
| **Deployment** | Vercel (frontend), Django (backend) |

---

## 5. Repository Structure

```
Theft_Sentinel/
├── theft_sentinel_backend/          # Django project root
│   ├── config/                     # Django settings, urls, MongoDB conn
│   │   ├── settings.py
│   │   ├── urls.py                 # Root router
│   │   └── mongodb.py
│   ├── apps/                       # Django apps
│   │   ├── accounts/               # Custom User + JWT auth + RBAC
│   │   ├── cameras/                # Camera model + MJPEG feed + RTSP proxy
│   │   ├── alerts/                 # Alert model + Cloudinary clip
│   │   ├── incidents/              # Incident management
│   │   ├── tracking/               # TrackingRecord (ingest only, UI removed)
│   │   ├── ai_engine/              # AI app: views, services, models
│   │   │   ├── api/views.py        # All AI HTTP views
│   │   │   └── services/           # continuous_monitor, inference_runner, ai_service, sse_registry, clip_encoding
│   │   ├── surveillance/           # AI event → Alert/Incident bridge
│   │   ├── dashboard/              # Aggregate stats endpoints
│   │   ├── feedback/               # Guard feedback
│   │   ├── personnel/              # Personnel management (admin)
│   │   ├── mobile/                 # SMS/Email notifications
│   │   ├── tenancy/                # Multi-tenancy
│   │   └── support/                # Support tickets
│   ├── ai_pipeline/                # Pure-Python ML modules (no Django ORM)
│   │   ├── ai_config/config.py     # All thresholds & hyper-parameters
│   │   ├── detection/              # YOLOv8 wrapper
│   │   ├── tracking/               # DeepSORT tracker
│   │   ├── reid/                   # OSNet feature extractor
│   │   ├── matching/               # FAISS matcher + identity DB
│   │   ├── theft_detection/        # X3D-S state machine (GlobalTheftDetector)
│   │   ├── x3d/                    # Compatibility shim → theft_detection
│   │   ├── camera/                 # Stream reader
│   │   ├── visualization/          # CV2 drawing
│   │   └── main.py                 # Standalone MCMT pipeline entry point
│   ├── manage.py
│   ├── .env                        # Environment variables
│   └── requirements.txt
│
├── theft-sentinel-frontend/         # React + Vite SPA
│   ├── src/
│   │   ├── pages/                  # auth, dashboard, cameras, alerts, incidents, ai, feedback, personnel, ...
│   │   ├── components/             # CameraFeedWithOverlay, LiveTrackingNodeGraph, Sidebar, ...
│   │   ├── router/AppRouter.jsx    # Role-protected routes
│   │   ├── hooks/                  # useRealtimeTracking, useAIEngine, ...
│   │   ├── api/                    # Axios client modules
│   │   └── store/                  # Recoil atoms
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── Updated_AI_Engine_v2/           # Standalone MCMT v2 reference implementation
├── THEFTSENTINEL_MCMT_DOCUMENTATION.md
├── AI_ENGINE_INTEGRATION.md
├── API_DOCUMENTATION.md
├── PROJECT_SUMMARY.md
├── INTEGRATION_PROMPT_FOR_AGENT.md
├── requirements.txt                # Top-level Python deps
└── vercel.json
```

---

## 6. Installation & Setup

### Prerequisites

- Python 3.10+ (with CUDA 12.x for GPU inference; CPU fallback supported)
- Node.js 18+
- MongoDB Atlas account (or local MongoDB)
- Cloudinary account (for alert clips)
- (Optional) Twilio account for SMS

### Backend Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd Theft_Sentinel

# 2. Create and activate virtual environment
cd theft_sentinel_backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Edit theft_sentinel_backend/.env with your MongoDB URI, Cloudinary, Twilio, email creds

# 5. Place the trained X3D checkpoint
# Copy best_model.pth into theft_sentinel_backend/

# 6. Run migrations (MongoDB backend)
python manage.py migrate

# 7. Start the backend server
python manage.py runserver
# Backend available at http://localhost:8000
```

### Frontend Setup

```bash
cd theft-sentinel-frontend

# 1. Install dependencies
npm install

# 2. (Optional) adjust API base URL
# Edit .env: VITE_API_BASE_URL=http://localhost:8000

# 3. Start dev server
npm run dev
# Frontend available at http://localhost:3000
```

### Standalone AI Pipeline (optional, for testing)

```bash
cd theft_sentinel_backend
python -m ai_pipeline.main --sources test_videos/cam1.mp4 test_videos/cam2.mp4
```

---

## 7. Usage & API Endpoints

### Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"..."}'
# Returns { access, refresh, user }
```

Use `Authorization: Bearer <access>` on all protected endpoints.

### Primary Endpoint Groups

| Group | Base Path | Key Endpoints |
|-------|-----------|---------------|
| Auth | `/api/auth/` | `register/`, `login/`, `refresh/`, `logout/` |
| Cameras | `/api/cameras/` | CRUD, `{id}/status/`, `{id}/feed/` (MJPEG, public) |
| Alerts | `/api/alerts/` | list, `{id}/`, `{id}/acknowledge/`, `active/` |
| Incidents | `/api/incidents/` | list, `{id}/assign/`, `{id}/status/` |
| Surveillance | `/api/surveillance/ingest/` | AI event → Alert/Incident bridge |
| Tracking | `/api/tracking/ingest/` | AI pipeline internal ingest |
| Mobile | `/api/mobile/` | `send-sms/`, `send-email/`, `send-bulk/` |
| Dashboard | `/api/dashboard/` | `overview/`, `alerts-stats/`, `incidents-stats/`, `recent-activity/`, `historical-reporting/` |
| Feedback | `/api/feedback/` | submit, `me/`, `stats/` |
| Personnel | `/api/personnel/` | CRUD (admin only) |
| AI Engine | `/api/ai/` | `analyze-frame/`, `process-camera/`, `full-pipeline/`, `monitor/start/`, `monitor/stop/`, `monitor/status/`, `cameras/{id}/realtime-tracking/` (SSE, public), `suspects/{id}/stop-tracking/`, `model-info/`, `health/` |

### Example: Start AI Monitoring

```bash
curl -X POST http://localhost:8000/api/ai/monitor/start/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"<camera_object_id>","restart":false}'
```

### Example: Consume SSE Stream (browser)

```javascript
const es = new EventSource(`/api/ai/cameras/${cameraId}/realtime-tracking/`);
es.onmessage = (e) => {
  const data = JSON.parse(e.data);
  // data.tracks[].bbox in native 640x480 coords; scale to canvas
};
```

### Example: Analyze a Single Frame

```bash
curl -X POST http://localhost:8000/api/ai/analyze-frame/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"frame":"<base64-jpeg>","camera_id":"<id>","create_alert_on_theft":true,"save_to_db":true}'
```

For the complete endpoint reference with payloads and responses, see `API_DOCUMENTATION.md`.

---

## 8. Evaluation & Future Scope

### Evaluation

- **Tracking accuracy**: Offline MOTA, IDF1, and ID-switch evaluation via `evaluate.py` (in `Updated_AI_Engine_v2/`) with ground-truth vs. prediction files and a threshold sweep.
- **Theft detection**: Confusion matrix on the Theft/normal test clips in `test_videos/Theft/`; rising-edge alert counting prevents per-frame inflation of metrics.
- **System health**: `/api/ai/health/` and `/api/ai/model-info/` report model load status and device; `/api/dashboard/overview/` aggregates camera/alert/incident stats.
- **Feedback loop**: Guard-submitted `FALSE_POSITIVE` / `TRUE_POSITIVE` feedback enables precision/recall tuning of `X3D_THEFT_THRESH` and `X3D_CONSECUTIVE_REQUIRED`.
- **Resource profiling**: Monitor RAM/VRAM with the documented `gc.collect()` / `torch.cuda.empty_cache()` cadence; target < 6 GB VRAM peak on a 6 GB GPU.

### Future Scope

1. **Multi-GPU / distributed inference** — shard cameras across GPUs; replace the process-level `inference_lock` with per-device locks and a load balancer.
2. **Active learning from guard feedback** — pipe `FALSE_POSITIVE`/`TRUE_POSITIVE` labels back into a retraining loop for the X3D-S checkpoint; add an admin UI for threshold tuning per camera.
3. **Edge deployment** — quantize YOLOv8 (TensorRT) and X3D (ONNX) for on-device inference at the camera, reducing bandwidth and centralizing only alerts.
4. **Pose-based theft heuristics** — the `DetectionTrack` model already stores behavioral features (`hand_in_bag_frames`, `concealment_events`); fuse these with X3D scores for a hybrid classifier.
5. **Real-time notification fan-out** — extend the SSE registry to push to mobile push (FCM/APNs) in addition to browser clients.

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| `PROJECT_SUMMARY.md` | Onboarding guide, DB schema, RBAC matrix, data flows |
| `API_DOCUMENTATION.md` | Full REST endpoint reference with payloads |
| `AI_ENGINE_INTEGRATION.md` | AI pipeline internals, optimizations, gotchas |
| `THEFTSENTINEL_MCMT_DOCUMENTATION.md` | Standalone MCMT module documentation |
| `INTEGRATION_PROMPT_FOR_AGENT.md` | X3D integration task specification |
| `Updated_AI_Engine_v2/README.md` | Standalone v2 pipeline quick start |

---

*Built with Django, React, PyTorch, and a lot of GPU memory management.*
