import cv2
import numpy as np
from collections import deque
import math

BLUE_LOWER = np.array([95, 100, 60])
BLUE_UPPER = np.array([135, 255, 255])

TENNIS_LOWER = np.array([24, 90, 80])
TENNIS_UPPER = np.array([45, 255, 255])

MIN_AREA_BLUE = 400
MIN_AREA_TENNIS = 300
MIN_CIRCULARITY_TENNIS = 0.65

TRAIL_LENGTH = 64

selected_blue_center = None
selection_lock = False

def select_blue_object(event, x, y, flags, param):
    global selected_blue_center, selection_lock
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_blue_center = (x, y)
        selection_lock = True
        print(f"[INFO] Locked onto Blue Object near ({x}, {y}). Press 'r' to reset lock.")


def preprocess_mask(hsv_frame, lower_bound, upper_bound, kernel_size=5):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_blue_object(mask):
    global selected_blue_center, selection_lock

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_BLUE:
            continue
            
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

    if selection_lock and selected_blue_center is not None:
        best_candidate = min(
            valid_candidates,
            key=lambda c: math.hypot(
                c["center"][0] - selected_blue_center[0],
                c["center"][1] - selected_blue_center[1]
            )
        )
        selected_blue_center = best_candidate["center"]
        return best_candidate

    best_candidate = max(valid_candidates, key=lambda c: c["area"])
    selected_blue_center = best_candidate["center"]
    return best_candidate


def detect_tennis_ball(mask):
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

        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY_TENNIS:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        circle_area = math.pi * (radius ** 2)

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

        frame = cv2.flip(frame, 1)

        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        blue_mask = preprocess_mask(hsv, BLUE_LOWER, BLUE_UPPER)
        tennis_mask = preprocess_mask(hsv, TENNIS_LOWER, TENNIS_UPPER)

        blue_obj = detect_blue_object(blue_mask)
        tennis_ball = detect_tennis_ball(tennis_mask)

        if blue_obj is not None:
            bx, by, bw, bh = blue_obj["bbox"]
            bcx, bcy = blue_obj["center"]
            b_area = blue_obj["area"]

            blue_center_trail.appendleft((bcx, bcy))

            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 140, 0), 2)
            cv2.drawContours(frame, [blue_obj["contour"]], -1, (255, 200, 0), 1)
            cv2.circle(frame, (bcx, bcy), 5, (255, 0, 0), -1)

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

        for i in range(1, len(blue_center_trail)):
            if blue_center_trail[i - 1] is None or blue_center_trail[i] is None:
                continue
            thickness = int(np.sqrt(TRAIL_LENGTH / float(i + 1)) * 2.2)
            cv2.line(frame, blue_center_trail[i - 1], blue_center_trail[i], (255, 100, 0), thickness )
        if tennis_ball is not None:
            tcx, tcy = tennis_ball["center"]
            t_radius = tennis_ball["radius"]
            t_area = tennis_ball["area"]

            cv2.circle(frame, (tcx, tcy), t_radius, (0, 255, 0), 2)
            cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)

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

        if blue_obj is not None and tennis_ball is not None:
            bcx, bcy = blue_obj["center"]
            tcx, tcy = tennis_ball["center"]

            dx = bcx - tcx
            dy = bcy - tcy
            distance = math.hypot(dx, dy)

            cv2.arrowedLine(frame, (tcx, tcy), (bcx, bcy), (255, 0, 255), 2, tipLength=0.04)

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
