import cv2
import numpy as np
import time
from collections import deque
from asset import detect_faces_mediapipe, gaze2point
from screeninfo import get_monitors

clicked_coords = []
calibration_retry_limit = 3
calibration_error_threshold = 100


def on_mouse_click(event, x, y, flags, param):
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords.append((x, y))
        print(f"Mouse clicked at: {x}, {y}")


def _get_primary_monitor():
    monitors = get_monitors()
    return next((m for m in monitors if getattr(m, "is_primary", False)), monitors[0])


def _get_primary_monitor_size():
    primary_monitor = _get_primary_monitor()
    return int(primary_monitor.width), int(primary_monitor.height)


def _get_window_canvas_size(window_name):
    fallback_w, fallback_h = _get_primary_monitor_size()
    init_canvas = np.ones((fallback_h, fallback_w, 3), dtype=np.uint8) * 255
    cv2.imshow(window_name, init_canvas)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)
    return fallback_w, fallback_h


def run_calibration(cap, estimator, visualizer, detector, posture_overlay=None):
    monitor = _get_primary_monitor()
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Calibration", int(monitor.x), int(monitor.y))
    cv2.resizeWindow("Calibration", int(monitor.width), int(monitor.height))
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    screen_width, screen_height = _get_window_canvas_size("Calibration")
    margin = max(40, int(min(screen_width, screen_height) * 0.08))

    calibration_coords = {
        "top_left": (margin, margin),
        "top_right": (screen_width - margin, margin),
        "bottom_left": (margin, screen_height - margin),
        "bottom_right": (screen_width - margin, screen_height - margin),
        "center": (screen_width // 2, screen_height // 2),
    }

    calibration_results = {}
    global clicked_coords
    clicked_coords = []

    cv2.setMouseCallback("Calibration", on_mouse_click)

    last_valid_gaze = (screen_width // 2, screen_height // 2)
    for point_name, coords in calibration_coords.items():
        # Retain the settled fixation, excluding the transition from the prior target.
        collected_points = deque(maxlen=12)
        clicked_coords.clear()

        while len(clicked_coords) == 0:
            canvas = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
            cv2.circle(canvas, coords, 20, (0, 0, 255), -1)
            cv2.putText(canvas, f"Look at the {point_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            status_text = "Faces: 0"
            status_color = (0, 0, 255)

            ret, frame = cap.read()
            if ret:
                faces = detect_faces_mediapipe(detector, frame)
                visualizer.set_image(frame)
                status_text = f"Faces: {len(faces)}"
                status_color = (0, 128, 0) if len(faces) > 0 else (0, 0, 255)
                for face in faces:
                    estimator.estimate(face, frame)
                    gx, gy = gaze2point(face.center * 1e3, face.gaze_vector)
                    collected_points.append((gx, gy))
                camera_view = posture_overlay.update(frame) if posture_overlay is not None else frame
                cv2.imshow("CalibrationCamera", camera_view)

            cv2.putText(canvas, status_text, (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.imshow("Calibration", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Calibration canceled by user.")
                cv2.destroyWindow("Calibration")
                try:
                    cv2.destroyWindow("CalibrationCamera")
                except Exception:
                    pass
                return None

        clicked_x, clicked_y = clicked_coords.pop(0)
        print(f"Calibration point {point_name}: Mouse clicked at {clicked_x}, {clicked_y}")

        if collected_points:
            avg_x = float(np.mean([p[0] for p in collected_points]))
            avg_y = float(np.mean([p[1] for p in collected_points]))
            last_valid_gaze = (avg_x, avg_y)
        else:
            avg_x, avg_y = last_valid_gaze
            print(f"[WARN] No gaze samples for {point_name}. Using last valid gaze.")

        distance = np.linalg.norm(np.array([avg_x, avg_y]) - np.array([clicked_x, clicked_y]))
        print(f"Distance between gaze and mouse at {point_name}: {distance:.2f} pixels")

        calibration_results[point_name] = {
            "gaze": (avg_x, avg_y),
            "mouse": (clicked_x, clicked_y),
        }

    cv2.destroyWindow("Calibration")
    try:
        cv2.destroyWindow("CalibrationCamera")
    except Exception:
        pass

    if len(calibration_results) == len(calibration_coords):
        print("Calibration successful for all points.")
        return calibration_results

    print("Calibration incomplete. Please retry.")
    return None


def calculate_diagonal(calibration_results):
    if not calibration_results:
        return None, None

    try:
        top_left_gaze = np.array(calibration_results["top_left"]["gaze"])
        bottom_right_gaze = np.array(calibration_results["bottom_right"]["gaze"])

        top_left_mouse = np.array(calibration_results["top_left"]["mouse"])
        bottom_right_mouse = np.array(calibration_results["bottom_right"]["mouse"])

        diagonal_gaze_pixels = np.linalg.norm(top_left_gaze - bottom_right_gaze)
        diagonal_mouse_pixels = np.linalg.norm(top_left_mouse - bottom_right_mouse)

        print(f"Diagonal length (gaze): {diagonal_gaze_pixels:.2f} pixels")
        print(f"Diagonal length (mouse): {diagonal_mouse_pixels:.2f} pixels")

        return diagonal_gaze_pixels, diagonal_mouse_pixels
    except KeyError as error:
        print(f"Error in calibration results: {error}")
        return None, None


def calculate_display_size(calibration_results, aspect_ratio_width=16, aspect_ratio_height=9):
    diagonal_gaze_pixels, diagonal_mouse_pixels = calculate_diagonal(calibration_results)
    if diagonal_gaze_pixels is None:
        return None, None, None, None

    aspect_ratio = np.sqrt(aspect_ratio_width ** 2 + aspect_ratio_height ** 2)
    width_ratio = aspect_ratio_width / aspect_ratio
    height_ratio = aspect_ratio_height / aspect_ratio

    width_gaze_pixels = diagonal_gaze_pixels * width_ratio
    height_gaze_pixels = diagonal_gaze_pixels * height_ratio

    width_mouse_pixels = diagonal_mouse_pixels * width_ratio
    height_mouse_pixels = diagonal_mouse_pixels * height_ratio

    print(f"Estimated width (gaze): {width_gaze_pixels:.2f} pixels")
    print(f"Estimated height (gaze): {height_gaze_pixels:.2f} pixels")
    print(f"Estimated width (mouse): {width_mouse_pixels:.2f} pixels")
    print(f"Estimated height (mouse): {height_mouse_pixels:.2f} pixels")

    return width_gaze_pixels, height_gaze_pixels, width_mouse_pixels, height_mouse_pixels
