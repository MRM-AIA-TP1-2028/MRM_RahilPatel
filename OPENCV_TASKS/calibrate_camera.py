"""
Webcam Calibration Script for VS Code
Detects a standard chessboard pattern from your webcam to compute:
- Camera Intrinsic Matrix (fx, fy, cx, cy)
- Lens Distortion Coefficients (k1, k2, p1, p2, k3)
Saves parameters to 'camera_calibration.json' for the ArUco tracker.
"""

import cv2
import numpy as np
import argparse
import json
import os


def generate_printable_chessboard(cols=9, rows=6, square_size_px=100, output_path="chessboard_pattern.png"):
    """Generates a printable chessboard pattern with specified inner corners."""
    board_cols, board_rows = cols + 1, rows + 1
    w, h = board_cols * square_size_px, board_rows * square_size_px
    image = np.ones((h, w), dtype=np.uint8) * 255

    for r in range(board_rows):
        for c in range(board_cols):
            if (r + c) % 2 == 1:
                y1, y2 = r * square_size_px, (r + 1) * square_size_px
                x1, x2 = c * square_size_px, (c + 1) * square_size_px
                image[y1:y2, x1:x2] = 0

    bordered = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(output_path, bordered)
    print(f"[INFO] Saved printable chessboard ({cols}x{rows} inner corners) to '{output_path}'.")


def run_calibration(
    camera_idx=0,
    checkerboard=(9, 6),
    square_size=2.5,
    min_samples=15,
    output_file="camera_calibration.json"
):
    cols, rows = checkerboard
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points = []

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam index {camera_idx}.")
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n" + "=" * 60)
    print("           WEBCAM CALIBRATION FOR ARUCO TRACKER")
    print("=" * 60)
    print(f"Target Pattern    : {cols} x {rows} inner corners")
    print(f"Square Edge Size  : {square_size} cm")
    print(f"Target Samples    : {min_samples}+ frames from different angles")
    print("-" * 60)
    print("Controls:")
    print("  [SPACE]         : Capture frame when corners turn green")
    print("  [C]             : Compute calibration matrix & save to JSON")
    print("  [Q] or [ESC]    : Exit without saving")
    print("=" * 60 + "\n")

    frame_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = gray.shape[::-1]
        display_frame = frame.copy()

        found, corners = cv2.findChessboardCorners(
            gray,
            checkerboard,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        sample_count = len(obj_points)
        status_color = (0, 255, 0) if found else (0, 0, 255)
        status_msg = "Pattern DETECTED! (Press SPACE)" if found else "Pattern not detected"

        if found:
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, checkerboard, refined_corners, found)

        h, w = display_frame.shape[:2]
        cv2.rectangle(display_frame, (0, 0), (w, 40), (20, 20, 20), -1)
        header_text = f"Samples: {sample_count}/{min_samples} | {status_msg}"
        cv2.putText(display_frame, header_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        help_bar = "[SPACE]: Capture Frame | [C]: Calculate & Save | [Q]: Quit"
        cv2.putText(display_frame, help_bar, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        cv2.imshow("Webcam Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            if found:
                obj_points.append(objp)
                img_points.append(refined_corners)
                print(f"[CAPTURE] Recorded sample #{len(obj_points)} / {min_samples}")
            else:
                print("[WARN] Chessboard must be fully visible with all corners detected before pressing SPACE.")

        elif key in (ord('c'), ord('C')):
            if len(obj_points) < 5:
                print(f"[WARN] Need at least 5 samples (have {len(obj_points)}). Recommended: 15-25.")
                continue

            print("\n[INFO] Running cv2.calibrateCamera optimizer...")
            ret_error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, frame_size, None, None
            )

            # Calculate reprojection error safely using NumPy
            total_error = 0.0
            for i in range(len(obj_points)):
                projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
                pts_real = img_points[i].reshape(-1, 2)
                pts_proj = projected.reshape(-1, 2)
                diff = np.linalg.norm(pts_real - pts_proj, axis=1)
                total_error += np.mean(diff)
            mean_error = float(total_error / len(obj_points))

            print("\n" + "=" * 60)
            print("                 CALIBRATION RESULTS")
            print("=" * 60)
            print(f"Reprojection Error (RMSE) : {mean_error:.4f} pixels (ideal < 0.5 px)")
            print(f"Focal Length (fx, fy)     : ({mtx[0,0]:.2f}, {mtx[1,1]:.2f})")
            print(f"Principal Point (cx, cy)  : ({mtx[0,2]:.2f}, {mtx[1,2]:.2f})")
            print(f"Distortion Coefficients   : {dist.ravel()[:5]}")
            print("=" * 60)

            calib_payload = {
                "camera_matrix": mtx.tolist(),
                "dist_coeffs": dist.tolist(),
                "image_width": frame_size[0],
                "image_height": frame_size[1],
                "reprojection_error": float(mean_error)
            }

            with open(output_file, "w") as f:
                json.dump(calib_payload, f, indent=4)

            print(f"\n[SUCCESS] Calibration saved to '{output_file}'!\n")
            break

        elif key in (27, ord('q'), ord('Q')):
            print("[INFO] Calibration aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera using a printed chessboard pattern.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument("--cols", type=int, default=9, help="Number of inner corners along width (default: 9).")
    parser.add_argument("--rows", type=int, default=6, help="Number of inner corners along height (default: 6).")
    parser.add_argument("--square-size", type=float, default=2.5, help="Square side length in cm (default: 2.5).")
    parser.add_argument("--output", type=str, default="camera_calibration.json", help="Output JSON filename.")
    parser.add_argument("--generate-pattern", action="store_true", help="Generate a printable chessboard image.")
    args = parser.parse_args()

    if args.generate_pattern:
        generate_printable_chessboard(cols=args.cols, rows=args.rows)
    else:
        run_calibration(
            camera_idx=args.camera,
            checkerboard=(args.cols, args.rows),
            square_size=args.square_size,
            output_file=args.output
        )
