# Real-Time Driver Monitoring System (DMS)

A computer vision-based Driver Monitoring System using Python, OpenCV, and MediaPipe on CPU.

## Roadmap (Phase 1)
- [x] **Step 1.1: Project Setup & OpenCV Webcam Feed**
- [x] **Step 1.2: MediaPipe Face Mesh Integration**
- [x] **Step 1.3: Eye Aspect Ratio (EAR) & Drowsiness Logic**
- [ ] **Step 1.4: 3D Head Pose Estimation & Distraction Logic**
- [ ] **Step 1.5: 5-Second Startup Calibration**

## Installation & Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Bash / Zsh
   source .venv/bin/activate.fish  # Fish shell
   # or: .venv\Scripts\activate    # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts:
   - Camera test (Step 1.1):
     ```bash
     python src/camera_test.py
     ```
   - Face Mesh test (Step 1.2):
     ```bash
     python src/face_mesh_test.py
     ```
   - Drowsiness Detection test (Step 1.3):
     ```bash
     python src/drowsiness_test.py
     ```
