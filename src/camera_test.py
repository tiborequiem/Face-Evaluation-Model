"""
Camera Test Script - Step 1.1 of Driver Monitoring System (DMS).

Opens the default webcam at 640x480 resolution, calculates and displays
real-time Frames Per Second (FPS), and exits cleanly when 'q' is pressed.
"""

import time
import cv2


def run_camera_test(camera_index: int = 0, width: int = 640, height: int = 480):
    """
    Initializes webcam feed, calculates live FPS, and displays the stream.

    Args:
        camera_index: Index of the camera device (0 is default built-in webcam).
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
    """
    print(f"[INFO] Connecting to camera device index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    # Set requested resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Verify if camera opened successfully
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with index {camera_index}.")
        print("[HINT] If using an external webcam, try index 1: run_camera_test(camera_index=1).")
        print("[HINT] Ensure camera permissions are granted and no other application is using it.")
        return

    # Actual negotiated resolution from camera driver
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened successfully. Resolution: {actual_width}x{actual_height}")
    print("[INFO] Press 'q' in the video window to exit.")

    # Variables for smoothed FPS calculation
    prev_time = time.time()
    fps_smooth = 0.0
    alpha = 0.9  # Smoothing factor: 0.9 retains historical average, 0.1 absorbs new instantaneous frame

    try:
        while True:
            # 1. Grab a single frame from the camera stream
            ret, frame = cap.read()

            # Verify that the frame was grabbed properly
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from camera stream. Exiting loop.")
                break

            # 2. Calculate FPS
            curr_time = time.time()
            delta_time = curr_time - prev_time
            prev_time = curr_time

            if delta_time > 0:
                instantaneous_fps = 1.0 / delta_time
                # Exponential Moving Average for a smooth, flicker-free readout
                if fps_smooth == 0.0:
                    fps_smooth = instantaneous_fps
                else:
                    fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * instantaneous_fps)

            # 3. Overlay information on the frame
            fps_text = f"FPS: {fps_smooth:.1f}"
            instructions_text = "Press 'q' to quit"

            # Draw background rectangle for readability
            cv2.rectangle(frame, (10, 10), (190, 70), (0, 0, 0), cv2.FILLED)

            # Draw text overlays
            cv2.putText(
                frame,
                fps_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green color in BGR format
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                instructions_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # 4. Display the frame in an interactive OpenCV window
            cv2.imshow("DMS - Step 1.1: Camera Feed & FPS", frame)

            # 5. Check for key press: waitKey(1) waits 1ms and flushes GUI events
            # 0xFF masks out non-ASCII bits for cross-platform compatibility
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] 'q' key pressed. Closing camera stream...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected. Closing camera stream...")

    finally:
        # 6. Clean up: Release camera hardware lock and destroy all GUI windows
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed cleanly.")


if __name__ == "__main__":
    run_camera_test(camera_index=0, width=640, height=480)
