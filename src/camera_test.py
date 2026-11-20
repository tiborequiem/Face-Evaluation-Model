import time
import cv2


def run_camera_test(camera_index=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Error: Unable to access camera at index {camera_index}")
        return

    prev_time = time.time()
    fps_smooth = 0.0
    alpha = 0.9
    is_running = True

    try:
        while is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Camera frame could not be retrieved.")
                break

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

            cv2.rectangle(frame, (10, 10), (190, 70), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"FPS: {fps_smooth:.1f}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("DMS - Step 1.1: Camera Feed & FPS", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                is_running = False

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_test()
