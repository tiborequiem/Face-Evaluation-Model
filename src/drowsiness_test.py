import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

MODEL_PATH = "models/face_landmarker.task"

# 6 landmarks per eye for EAR calculation
# Format: [P1_corner, P2_top1, P3_top2, P4_corner, P5_bot2, P6_bot1]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.23
DROWSY_CONSECUTIVE_FRAMES = 25


def create_landmarker():
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(landmarks, indices, width, height):
    pts = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in indices]

    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0, pts

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear, pts


def draw_ear_indicators(frame, pts, color):
    # Draw EAR measurement lines (horizontal and 2 vertical)
    p1, p2, p3, p4, p5, p6 = pts
    cv2.line(frame, p1, p4, color, 1, cv2.LINE_AA)
    cv2.line(frame, p2, p6, color, 1, cv2.LINE_AA)
    cv2.line(frame, p3, p5, color, 1, cv2.LINE_AA)

    # Highlight the 6 landmark dots
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1, cv2.LINE_AA)


def run_drowsiness_test(camera_index=0, width=640, height=480):
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

    closed_frame_count = 0
    is_drowsy = False
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

            avg_ear = 0.0
            face_detected = False

            if detection_result.face_landmarks:
                face_detected = True
                landmarks = detection_result.face_landmarks[0]

                right_ear, right_pts = calculate_ear(landmarks, RIGHT_EYE_INDICES, w, h)
                left_ear, left_pts = calculate_ear(landmarks, LEFT_EYE_INDICES, w, h)
                avg_ear = (right_ear + left_ear) / 2.0

                # Determine eye state
                eye_color = (0, 0, 255) if avg_ear < EAR_THRESHOLD else (0, 255, 0)
                draw_ear_indicators(frame, right_pts, eye_color)
                draw_ear_indicators(frame, left_pts, eye_color)

                # Drowsiness frame counter logic
                if avg_ear < EAR_THRESHOLD:
                    closed_frame_count += 1
                    if closed_frame_count >= DROWSY_CONSECUTIVE_FRAMES:
                        is_drowsy = True
                else:
                    closed_frame_count = 0
                    is_drowsy = False
            else:
                # If no face is detected, reset drowsy state
                closed_frame_count = 0
                is_drowsy = False

            # Draw HUD panel
            cv2.rectangle(frame, (10, 10), (280, 120), (0, 0, 0), cv2.FILLED)
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

            ear_color = (0, 0, 255) if avg_ear < EAR_THRESHOLD and face_detected else (0, 255, 0)
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.2f} (Thresh: {EAR_THRESHOLD})",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                ear_color,
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Closed Frames: {closed_frame_count}/{DROWSY_CONSECUTIVE_FRAMES}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Press 'q' to quit",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # Driver state banner
            if is_drowsy:
                cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), cv2.FILLED)
                cv2.putText(
                    frame,
                    "** DROWSINESS ALERT! WAKE UP! **",
                    (w // 2 - 200, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif face_detected:
                cv2.rectangle(frame, (0, h - 35), (w, h), (0, 180, 0), cv2.FILLED)
                cv2.putText(
                    frame,
                    "STATUS: DRIVER ALERT (NORMAL)",
                    (w // 2 - 140, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("DMS - Step 1.3: Drowsiness Detection (EAR)", frame)

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
    run_drowsiness_test()
