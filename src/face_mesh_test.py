import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

MODEL_PATH = "models/face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

KEY_LANDMARK_INDICES = {
    "Nose Tip": 1,
    "Chin": 152,
    "Right Eye Outer": 33,
    "Right Eye Inner": 133,
    "Left Eye Outer": 263,
    "Left Eye Inner": 362,
    "Right Mouth Corner": 61,
    "Left Mouth Corner": 291,
}


def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading face landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")


def create_landmarker():
    ensure_model_file()
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def draw_eye_contours(frame, landmarks, width, height):
    left_connections = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
    right_connections = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE

    for conn in list(left_connections) + list(right_connections):
        start = landmarks[conn.start]
        end = landmarks[conn.end]
        pt1 = (int(start.x * width), int(start.y * height))
        pt2 = (int(end.x * width), int(end.y * height))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)


def draw_key_reference_points(frame, landmarks, width, height):
    for _, idx in KEY_LANDMARK_INDICES.items():
        lm = landmarks[idx]
        pt = (int(lm.x * width), int(lm.y * height))
        cv2.circle(frame, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)


def run_face_mesh_test(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Error: Unable to access camera at index {camera_index}")
        return

    landmarker = create_landmarker()

    prev_time = time.time()
    last_timestamp_ms = 0
    fps_smooth = 0.0
    alpha = 0.9
    is_running = True

    try:
        while is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Camera frame could not be retrieved.")
                break

            h, w, _ = frame.shape

            curr_time = time.time()
            delta = curr_time - prev_time
            prev_time = curr_time

            if delta > 0:
                instant_fps = 1.0 / delta
                fps_smooth = (
                    instant_fps
                    if fps_smooth == 0.0
                    else (alpha * fps_smooth) + ((1.0 - alpha) * instant_fps)
                )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(curr_time * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            face_detected = False
            if detection_result.face_landmarks:
                face_detected = True
                landmarks = detection_result.face_landmarks[0]
                draw_eye_contours(frame, landmarks, w, h)
                draw_key_reference_points(frame, landmarks, w, h)

            cv2.rectangle(frame, (10, 10), (220, 85), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"FPS: {fps_smooth:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            status_color = (0, 255, 0) if face_detected else (0, 0, 255)
            status_text = "Face: Detected" if face_detected else "Face: Not Found"
            cv2.putText(
                frame,
                status_text,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Press 'q' to quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("DMS - Step 1.2: MediaPipe Face Mesh", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                is_running = False

    except KeyboardInterrupt:
        pass
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_mesh_test()
