"""
Real-Time ArUco Marker Detection, Pose Estimation, and Depth Measurement
Optimized for VS Code & Python (OpenCV 4.x compatible).
Detects 4x4, 5x5, and 6x6 ArUco markers simultaneously.
Tags can be generated from: https://chev.me/arucogen
"""

import cv2
import numpy as np
import argparse
import time
import os
import sys
import json


ARUCO_DICT_MAPPING = {
    # 4x4 dictionaries
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    # 5x5 dictionaries
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    # 6x6 dictionaries
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
}

MULTI_DICT_KEYS = ["DICT_4X4_250", "DICT_5X5_250", "DICT_6X6_250"]


def has_gui_display():
    """Checks if a graphical desktop window can be displayed."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_default_camera_matrix(width, height):
    """Approximates focal length and principal point if no calibration file exists."""
    focal_length = float(width)
    cx, cy = float(width) / 2.0, float(height) / 2.0
    camera_matrix = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def load_camera_calibration(calib_file):
    """Loads camera calibration from JSON if available."""
    if not calib_file or not os.path.exists(calib_file):
        return None, None
    try:
        with open(calib_file, "r") as f:
            data = json.load(f)
        return np.array(data["camera_matrix"], dtype=np.float64), np.array(data["dist_coeffs"], dtype=np.float64)
    except Exception as e:
        print(f"[WARN] Error reading '{calib_file}': {e}")
        return None, None


def create_aruco_detector(dict_enum):
    """Creates detector instance compatible with OpenCV >= 4.7 and older versions."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_enum)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return ("new", detector, dictionary, parameters)
    else:
        return ("legacy", None, dictionary, parameters)


def detect_markers(image, detector_pack):
    """Detects marker corners and IDs."""
    mode, detector, dictionary, parameters = detector_pack
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if mode == "new":
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids


def estimate_pose_single_marker(corners, marker_size, camera_matrix, dist_coeffs):
    """Estimates 3D pose using IPPE square planar solver."""
    half_size = marker_size / 2.0
    obj_points = np.array([
        [-half_size,  half_size, 0.0],
        [ half_size,  half_size, 0.0],
        [ half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0]
    ], dtype=np.float64)

    image_points = corners.reshape((4, 2)).astype(np.float64)
    return cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)


def rotation_to_euler_angles(rvec):
    """Converts Rodrigues vector to Roll, Pitch, Yaw in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    return angles[2], angles[0], angles[1]  # roll, pitch, yaw


def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, length):
    """Draws 3D axes (X: Red, Y: Green, Z: Blue)."""
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length, 2)
    elif hasattr(cv2.aruco, "drawAxis"):
        cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, length)


def draw_info_overlay(frame, detections_info, mode_name, fps, calib_status, show_hud=True):
    """Overlays telemetry panels and badges onto the frame."""
    if not show_hud:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    header = f"ArUco Tracker | Mode: {mode_name} | FPS: {fps:.1f} | Calib: {calib_status} | Found: {len(detections_info)}"
    cv2.putText(frame, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

    card_y = 48
    for info in detections_info:
        marker_id = info["id"]
        dict_label = info["dict_name"]
        x, y, z = info["position"]
        roll, pitch, yaw = info["orientation"]
        dist = info["distance"]
        cx, cy = info["center"]

        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{marker_id} ({dict_label})", (int(cx) + 12, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Z:{z:.1f}cm | D:{dist:.1f}cm", (int(cx) + 12, int(cy) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

        panel_w, panel_h = 330, 92
        if card_y + panel_h < h - 35:
            panel_overlay = frame.copy()
            cv2.rectangle(panel_overlay, (10, card_y), (10 + panel_w, card_y + panel_h), (10, 10, 10), -1)
            cv2.addWeighted(panel_overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (10, card_y), (10 + panel_w, card_y + panel_h), (90, 90, 90), 1)

            cv2.putText(frame, f"Marker #{marker_id} [{dict_label}]", (20, card_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Pos (cm): X={x:+.1f}  Y={y:+.1f}  Z={z:+.1f}", (20, card_y + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Rot (deg): R={roll:+.1f}  P={pitch:+.1f}  Y={yaw:+.1f}", (20, card_y + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Depth (Z): {z:.1f} cm  |  Direct: {dist:.1f} cm", (20, card_y + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
            card_y += panel_h + 8

    help_bar = "[Q]: Quit | [A]: Multi-Dict | [4]: 4x4 | [5]: 5x5 | [6]: 6x6 | [H]: HUD | [S]: Save"
    cv2.putText(frame, help_bar, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def process_frame(frame, active_detectors, marker_size, camera_matrix, dist_coeffs):
    """Detects markers and calculates 3D pose across dictionaries."""
    detections_info = []

    for dict_name, detector_pack in active_detectors.items():
        corners, ids = detect_markers(frame, detector_pack)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                c = corners[i]
                success, rvec, tvec = estimate_pose_single_marker(
                    c, marker_size, camera_matrix, dist_coeffs
                )

                if success:
                    draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.75)
                    tx, ty, tz = tvec.flatten()
                    distance = np.sqrt(tx**2 + ty**2 + tz**2)
                    roll, pitch, yaw = rotation_to_euler_angles(rvec)

                    pts = c.reshape((4, 2))
                    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

                    detections_info.append({
                        "id": int(ids[i][0]),
                        "dict_name": dict_name.replace("DICT_", ""),
                        "position": (float(tx), float(ty), float(tz)),
                        "orientation": (float(roll), float(pitch), float(yaw)),
                        "distance": float(distance),
                        "center": (cx, cy),
                        "rvec": rvec,
                        "tvec": tvec
                    })

    return frame, detections_info


def generate_sample_markers(output_dir="sample_markers"):
    """Creates sample 4x4, 5x5, and 6x6 marker images."""
    os.makedirs(output_dir, exist_ok=True)
    samples = [
        ("DICT_4X4_250", 12, "marker_4x4_id12.png"),
        ("DICT_5X5_250", 34, "marker_5x5_id34.png"),
        ("DICT_6X6_250", 56, "marker_6x6_id56.png"),
    ]
    created = []
    for dict_key, marker_id, filename in samples:
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAPPING[dict_key])
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, 300)
        bordered = cv2.copyMakeBorder(marker_img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, bordered)
        created.append(path)
    print(f"[INFO] Sample markers created in '{output_dir}': {created}")
    return created


def main():
    parser = argparse.ArgumentParser(description="Real-time ArUco Pose & Depth Estimator.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--dict", type=str, default="ALL", help="'ALL', '4X4_250', '5X5_250', '6X6_250'.")
    parser.add_argument("--marker-size", type=float, default=5.0, help="Marker edge size in cm (default: 5.0).")
    parser.add_argument("--calib", type=str, default="camera_calibration.json", help="Path to calibration file.")
    parser.add_argument("--image", type=str, default=None, help="Process a static image file.")
    parser.add_argument("--generate-markers", action="store_true", help="Generate sample markers.")
    args = parser.parse_args()

    if args.generate_markers:
        generate_sample_markers()
        return

    all_detectors = {k: create_aruco_detector(v) for k, v in ARUCO_DICT_MAPPING.items()}

    def select_active_detectors(mode_str):
        mode_upper = mode_str.upper()
        if mode_upper in ("ALL", "MULTI", "AUTO"):
            return {k: all_detectors[k] for k in MULTI_DICT_KEYS}, "ALL (4x4, 5x5, 6x6)"
        elif "4X4" in mode_upper:
            target = "DICT_4X4_250" if mode_upper == "4X4" else (f"DICT_{mode_upper}" if not mode_upper.startswith("DICT_") else mode_upper)
            target = target if target in all_detectors else "DICT_4X4_250"
            return {target: all_detectors[target]}, target.replace("DICT_", "")
        elif "5X5" in mode_upper:
            target = "DICT_5X5_250" if mode_upper == "5X5" else (f"DICT_{mode_upper}" if not mode_upper.startswith("DICT_") else mode_upper)
            target = target if target in all_detectors else "DICT_5X5_250"
            return {target: all_detectors[target]}, target.replace("DICT_", "")
        elif "6X6" in mode_upper:
            target = "DICT_6X6_250" if mode_upper == "6X6" else (f"DICT_{mode_upper}" if not mode_upper.startswith("DICT_") else mode_upper)
            target = target if target in all_detectors else "DICT_6X6_250"
            return {target: all_detectors[target]}, target.replace("DICT_", "")
        else:
            return {k: all_detectors[k] for k in MULTI_DICT_KEYS}, "ALL (4x4, 5x5, 6x6)"

    active_detectors, current_mode_name = select_active_detectors(args.dict)
    show_hud = True
    gui_available = has_gui_display()

    # Static image test mode
    if args.image is not None:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[ERROR] Could not load image: {args.image}")
            return
        h, w = frame.shape[:2]
        camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
        calib_status = "Loaded (JSON)" if camera_matrix is not None else "Estimated"
        if camera_matrix is None:
            camera_matrix, dist_coeffs = get_default_camera_matrix(w, h)

        annotated, detections = process_frame(frame, active_detectors, args.marker_size, camera_matrix, dist_coeffs)
        annotated = draw_info_overlay(annotated, detections, current_mode_name, fps=30.0, calib_status=calib_status, show_hud=True)

        print(f"\n[RESULTS] Detected {len(detections)} marker(s):")
        for d in detections:
            tx, ty, tz = d["position"]
            r, p, y = d["orientation"]
            print(f"  - ID {d['id']} [{d['dict_name']}]: Pos=({tx:+.1f}, {ty:+.1f}, {tz:+.1f}) cm | Depth={tz:.1f} cm | Dist={d['distance']:.1f} cm")

        out_path = "aruco_detection_result.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Saved output to '{out_path}'.")

        if gui_available:
            cv2.imshow("ArUco Pose Estimation", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Real-time webcam mode
    print(f"\n[INFO] Opening webcam (index {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera initialized at {actual_w}x{actual_h}")

    camera_matrix, dist_coeffs = load_camera_calibration(args.calib)
    if camera_matrix is not None:
        calib_status = "Custom (JSON)"
        print(f"[INFO] Loaded calibration from '{args.calib}'.")
    else:
        calib_status = "Estimated"
        print(f"[INFO] '{args.calib}' not found. Using estimated intrinsics.")
        camera_matrix, dist_coeffs = get_default_camera_matrix(actual_w, actual_h)

    print(f"[INFO] Marker size: {args.marker_size} cm")
    print(f"[INFO] Tracking mode: {current_mode_name}")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            annotated, detections = process_frame(frame, active_detectors, args.marker_size, camera_matrix, dist_coeffs)
            annotated = draw_info_overlay(
                annotated, detections, current_mode_name, fps=fps, calib_status=calib_status, show_hud=show_hud
            )

            if gui_available:
                cv2.imshow("ArUco Marker Pose & Depth Tracker", annotated)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('a'), ord('A')):
                active_detectors, current_mode_name = select_active_detectors("ALL")
            elif key == ord('4'):
                active_detectors, current_mode_name = select_active_detectors("4X4")
            elif key == ord('5'):
                active_detectors, current_mode_name = select_active_detectors("5X5")
            elif key == ord('6'):
                active_detectors, current_mode_name = select_active_detectors("6X6")
            elif key in (ord('h'), ord('H')):
                show_hud = not show_hud
            elif key in (ord('s'), ord('S')):
                snap_name = f"aruco_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, annotated)
                print(f"[INFO] Snapshot saved to '{snap_name}'.")

    finally:
        cap.release()
        if gui_available:
            cv2.destroyAllWindows()
        print("[INFO] Camera released. Program exited.")


if __name__ == "__main__":
    main()