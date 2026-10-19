import cv2
import numpy as np
from collections import deque
import math


# CONFIGURATION & DEFAULT HSV THRESHOLDS

# Default HSV range for Blue Object
BLUE_LOWER = np.array([95, 100, 60])
BLUE_UPPER = np.array([135, 255, 255])

# Default HSV range for Tennis Ball (Optic Yellow-Green)
TENNIS_LOWER = np.array([24, 90, 80])
TENNIS_UPPER = np.array([45, 255, 255])

# Minimum contour area (in px^2) to filter out camera sensor noise
MIN_AREA_BLUE = 400
MIN_AREA_TENNIS = 300
MIN_CIRCULARITY_TENNIS = 0.65

# Maximum number of points stored to draw the blue object's center tracking trail
TRAIL_LENGTH = 64

# State variables for optional interactive mouse selection of a specific blue object
selected_blue_center = None
selection_lock = False


def select_blue_object(event, x, y, flags, param):
    """Mouse callback to select a specific blue object by clicking on it."""
    global selected_blue_center, selection_lock
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_blue_center = (x, y)
        selection_lock = True
        print(f"[INFO] Locked onto Blue Object near ({x}, {y}). Press 'r' to reset lock.")


def preprocess_mask(hsv_frame, lower_bound, upper_bound, kernel_size=5):
    """Creates a clean binary mask using morphological opening and closing."""
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Opening removes small background noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Closing fills small holes inside the detected foreground object
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_blue_object(mask):
    """
    Detects the blue object without assuming a circular shape.
    If the user clicked a specific blue object, tracks the contour closest to the last known position.
    Otherwise, tracks the largest valid blue contour.
    """
    global selected_blue_center, selection_lock

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_BLUE:
            continue

        # Compute true centroid using image moments (works for any irregular shape)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        bbox = cv2.boundingRect(cnt)  # (x, y, w, h)
        valid_candidates.append({
            "contour": cnt,
            "area": area,
            "center": (cx, cy),
            "bbox": bbox
        })

    if not valid_candidates:
        return None

    # If user clicked a specific blue object, lock onto the contour nearest to the previous centroid
    if selection_lock and selected_blue_center is not None:
        best_candidate = min(
            valid_candidates,
            key=lambda c: math.hypot(
                c["center"][0] - selected_blue_center[0],
                c["center"][1] - selected_blue_center[1]
            )
        )
        # Update the tracked center for the next frame
        selected_blue_center = best_candidate["center"]
        return best_candidate

    # Default: return the largest blue object by area
    best_candidate = max(valid_candidates, key=lambda c: c["area"])
    selected_blue_center = best_candidate["center"]
    return best_candidate


def detect_tennis_ball(mask):
    """
    Detects the tennis ball using both HSV color segmentation and circularity filtering.
    Returns center (x, y), radius (px), and area (px^2).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_ball = None
    max_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_TENNIS:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Circularity = 4 * pi * Area / Perimeter^2 (1.0 is a perfect circle)
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY_TENNIS:
            continue

        # Fit minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # Circle area in px^2 (also keep contour area if needed)
        circle_area = math.pi * (radius ** 2)

        # Rank candidate by area * circularity
        score = area * circularity
        if score > max_score:
            max_score = score
            best_ball = {
                "center": center,
                "radius": radius,
                "area": circle_area,
                "contour_area": area
            }

    return best_ball


def main():
    global selection_lock, selected_blue_center

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open live camera stream.")
        return

    # Deque to store historical positions of the blue object's center
    blue_center_trail = deque(maxlen=TRAIL_LENGTH)

    window_name = "Blue Object & Tennis Ball Tracker"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_blue_object)

    print("=== Controls ===")
    print(" - Left-click on any blue object to lock tracking onto it.")
    print(" - Press 'r' to reset selection lock and clear trajectory trail.")
    print(" - Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            break

        # Mirror frame horizontally for intuitive live camera interaction
        frame = cv2.flip(frame, 1)

        # Apply Gaussian blur to reduce high-frequency camera noise
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 1. Create Binary Masks
        blue_mask = preprocess_mask(hsv, BLUE_LOWER, BLUE_UPPER)
        tennis_mask = preprocess_mask(hsv, TENNIS_LOWER, TENNIS_UPPER)

        # 2. Detect Objects Independently
        blue_obj = detect_blue_object(blue_mask)
        tennis_ball = detect_tennis_ball(tennis_mask)

        # 3. Render Blue Object Output
        if blue_obj is not None:
            bx, by, bw, bh = blue_obj["bbox"]
            bcx, bcy = blue_obj["center"]
            b_area = blue_obj["area"]

            # Append center to tracking trail
            blue_center_trail.appendleft((bcx, bcy))

            # Draw Bounding Box (Non-circular shape assumption)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 140, 0), 2)
            # Draw exact contour outline for arbitrary geometry visualization
            cv2.drawContours(frame, [blue_obj["contour"]], -1, (255, 200, 0), 1)
            # Draw Center Point
            cv2.circle(frame, (bcx, bcy), 5, (255, 0, 0), -1)

            # Annotate Blue Object Metrics
            lock_status = " [LOCKED]" if selection_lock else ""
            cv2.putText(
                frame,
                f"Blue Object{lock_status}",
                (bx, max(by - 28, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2
            )
            cv2.putText(
                frame,
                f"Area: {b_area:.1f} px^2 | Center: ({bcx}, {bcy})",
                (bx, max(by - 8, 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )
        else:
            # Gradually clear trail if object leaves frame
            if len(blue_center_trail) > 0:
                blue_center_trail.pop()

        # Draw historical line tracking the Blue Object's center
        for i in range(1, len(blue_center_trail)):
            if blue_center_trail[i - 1] is None or blue_center_trail[i] is None:
                continue
            thickness = int(np.sqrt(TRAIL_LENGTH / float(i + 1)) * 2.2)
            cv2.line(frame, blue_center_trail[i - 1], blue_center_trail[i], (255, 100, 0), thickness)

        # 4. Render Tennis Ball Output
        if tennis_ball is not None:
            tcx, tcy = tennis_ball["center"]
            t_radius = tennis_ball["radius"]
            t_area = tennis_ball["area"]

            # Draw Enclosing Circle and Center
            cv2.circle(frame, (tcx, tcy), t_radius, (0, 255, 0), 2)
            cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)

            # Annotate Tennis Ball Metrics
            label_y = max(tcy - t_radius - 30, 25)
            cv2.putText(
                frame,
                f"Tennis Ball | Center: ({tcx}, {tcy})",
                (max(tcx - t_radius, 10), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Radius: {t_radius} px | Area: {t_area:.1f} px^2",
                (max(tcx - t_radius, 10), label_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # 5. Relative Position of Blue Object wrt Tennis Ball (if both exist)
        if blue_obj is not None and tennis_ball is not None:
            bcx, bcy = blue_obj["center"]
            tcx, tcy = tennis_ball["center"]

            # Relative coordinates of Blue Object wrt Tennis Ball origin
            dx = bcx - tcx
            dy = bcy - tcy
            distance = math.hypot(dx, dy)

            # Draw connecting vector line from Tennis Ball to Blue Object
            cv2.arrowedLine(frame, (tcx, tcy), (bcx, bcy), (255, 0, 255), 2, tipLength=0.04)

            # Midpoint for annotation
            mid_x, mid_y = (bcx + tcx) // 2, (bcy + tcy) // 2
            cv2.putText(
                frame,
                f"Rel Pos wrt Ball: dx={dx:+d}px, dy={dy:+d}px (Dist: {distance:.1f}px)",
                (max(mid_x - 140, 10), max(mid_y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2
            )

        # Display output window
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            selection_lock = False
            selected_blue_center = None
            blue_center_trail.clear()
            print("[INFO] Reset object selection lock and trail.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
