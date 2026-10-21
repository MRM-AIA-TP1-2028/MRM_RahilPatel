import cv2
import numpy as np
from collections import deque
import math

BLUE_LOWER = np.array([95, 100, 60])
BLUE_UPPER = np.array([135, 255, 255])

TENNIS_LOWER = np.array([22, 90, 80])
TENNIS_UPPER = np.array([45, 255, 255])

MIN_AREA_BLUE = 400
MIN_AREA_TENNIS = 300
MIN_CIRCULARITY_TENNIS = 0.65

TRAIL_LENGTH = 50
MAX_MATCH_DISTANCE = 120.0
MAX_DISAPPEARED_FRAMES = 15
JITTER_THRESHOLD = 2.0


class MultiBlueObjectTracker:
    def __init__(self):
        self.next_id = 1
        self.objects = {}

    def reset(self):
        self.next_id = 1
        self.objects.clear()

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]["disappeared"] += 1
                if self.objects[obj_id]["disappeared"] > MAX_DISAPPEARED_FRAMES:
                    del self.objects[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return self.objects

        obj_ids = list(self.objects.keys())
        existing_centers = [self.objects[oid]["center"] for oid in obj_ids]

        dist_matrix = np.zeros((len(existing_centers), len(detections)), dtype=np.float32)
        for i, (ex, ey) in enumerate(existing_centers):
            for j, det in enumerate(detections):
                dx = ex - det["center"][0]
                dy = ey - det["center"][1]
                dist_matrix[i, j] = math.hypot(dx, dy)

        matched_rows = set()
        matched_cols = set()

        flat_indices = np.argsort(dist_matrix, axis=None)
        for flat_idx in flat_indices:
            row = int(flat_idx // len(detections))
            col = int(flat_idx % len(detections))

            if row in matched_rows or col in matched_cols:
                continue
            if dist_matrix[row, col] > MAX_MATCH_DISTANCE:
                continue

            obj_id = obj_ids[row]
            det = detections[col]

            prev_cx, prev_cy = self.objects[obj_id]["center"]
            curr_cx, curr_cy = det["center"]
            step_dist = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)

            if step_dist >= JITTER_THRESHOLD:
                self.objects[obj_id]["total_distance"] += step_dist

            self.objects[obj_id]["center"] = det["center"]
            self.objects[obj_id]["area"] = det["area"]
            self.objects[obj_id]["contour"] = det["contour"]
            self.objects[obj_id]["trail"].appendleft(det["center"])
            self.objects[obj_id]["disappeared"] = 0

            matched_rows.add(row)
            matched_cols.add(col)

        for row, obj_id in enumerate(obj_ids):
            if row not in matched_rows:
                self.objects[obj_id]["disappeared"] += 1
                if self.objects[obj_id]["disappeared"] > MAX_DISAPPEARED_FRAMES:
                    del self.objects[obj_id]

        for col, det in enumerate(detections):
            if col not in matched_cols:
                self._register(det)

        return self.objects

    def _register(self, det):
        trail = deque(maxlen=TRAIL_LENGTH)
        trail.appendleft(det["center"])
        self.objects[self.next_id] = {
            "center": det["center"],
            "area": det["area"],
            "contour": det["contour"],
            "trail": trail,
            "total_distance": 0.0,
            "disappeared": 0
        }
        self.next_id += 1


def preprocess_mask(hsv_frame, lower_bound, upper_bound, kernel_size=5):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_blue_objects(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_BLUE:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        detections.append({
            "contour": cnt,
            "area": area,
            "center": (cx, cy)
        })

    return detections


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
                "area": circle_area
            }

    return best_ball


def render_black_dashboard(tracked_blue_objects, tennis_ball):
    active_objects = {
        oid: data for oid, data in tracked_blue_objects.items() if data["disappeared"] == 0
    }

    width = 820
    row_height = 95
    height = max(480, 140 + len(active_objects) * row_height)

    dashboard = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(
        dashboard,
        "BLUE OBJECTS TELEMETRY DASHBOARD",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 200, 0),
        2
    )

    if tennis_ball is not None:
        tx, ty = tennis_ball["center"]
        t_info = (
            f"Yellow/Tennis Ball: Center=({tx}, {ty}) px | "
            f"Radius={tennis_ball['radius']} px | Area={tennis_ball['area']:.1f} px^2"
        )
        t_color = (0, 255, 100)
    else:
        t_info = "Yellow/Tennis Ball: NOT DETECTED IN FRAME"
        t_color = (0, 100, 255)

    cv2.putText(dashboard, t_info, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_color, 1)
    cv2.line(dashboard, (20, 85), (width - 20, 85), (80, 80, 80), 1)

    if not active_objects:
        cv2.putText(
            dashboard,
            "No Blue Objects Currently Detected...",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (160, 160, 160),
            1
        )
        return dashboard

    y_offset = 120
    for obj_id, data in sorted(active_objects.items()):
        cx, cy = data["center"]
        area = data["area"]
        traversed = data["total_distance"]

        line1 = (
            f"Blue Object #{obj_id}  |  (x, y): ({cx}, {cy}) px  |  "
            f"Area: {area:.1f} px^2"
        )
        cv2.putText(
            dashboard,
            line1,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2
        )

        if tennis_ball is not None:
            tx, ty = tennis_ball["center"]
            dx = cx - tx
            dy = cy - ty
            rel_dist = math.hypot(dx, dy)
            line2 = (
                f"   -> Pos wrt Yellow Ball: dx={dx:+d} px, dy={dy:+d} px "
                f"(Distance: {rel_dist:.1f} px)"
            )
            rel_color = (255, 120, 255)
        else:
            line2 = "   -> Pos wrt Yellow Ball: N/A (Yellow ball not in frame)"
            rel_color = (150, 150, 150)

        cv2.putText(
            dashboard,
            line2,
            (20, y_offset + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            rel_color,
            1
        )

        line3 = f"   -> Distance Traversed:  {traversed:.1f} px"
        cv2.putText(
            dashboard,
            line3,
            (20, y_offset + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 255, 255),
            1
        )

        cv2.line(
            dashboard,
            (20, y_offset + 68),
            (width - 20, y_offset + 68),
            (45, 45, 45),
            1
        )
        y_offset += row_height

    return dashboard


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open live camera stream.")
        return

    blue_tracker = MultiBlueObjectTracker()

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

        blue_detections = detect_blue_objects(blue_mask)
        tracked_blue = blue_tracker.update(blue_detections)
        tennis_ball = detect_tennis_ball(tennis_mask)

        if tennis_ball is not None:
            tcx, tcy = tennis_ball["center"]
            t_radius = tennis_ball["radius"]
            t_area = tennis_ball["area"]

            cv2.circle(frame, (tcx, tcy), t_radius, (0, 255, 0), 2)
            cv2.circle(frame, (tcx, tcy), 5, (0, 0, 255), -1)

            label_y = max(tcy - t_radius - 25, 20)
            cv2.putText(
                frame,
                f"Tennis Ball ({tcx},{tcy}) R:{t_radius}px Area:{t_area:.0f}px^2",
                (max(tcx - t_radius, 10), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        for obj_id, data in tracked_blue.items():
            if data["disappeared"] > 0:
                continue

            bcx, bcy = data["center"]
            b_area = data["area"]
            contour = data["contour"]
            trail = data["trail"]

            cv2.drawContours(frame, [contour], -1, (255, 180, 0), 2)
            cv2.circle(frame, (bcx, bcy), 5, (0, 0, 255), -1)

            for i in range(1, len(trail)):
                if trail[i - 1] is None or trail[i] is None:
                    continue
                thickness = int(np.sqrt(TRAIL_LENGTH / float(i + 1)) * 2.0)
                cv2.line(frame, trail[i - 1], trail[i], (255, 100, 0), thickness)

            cv2.putText(
                frame,
                f"Blue #{obj_id} | Area: {b_area:.0f}px^2",
                (max(bcx - 50, 10), max(bcy - 15, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 220, 0),
                2
            )

            if tennis_ball is not None:
                tcx, tcy = tennis_ball["center"]
                cv2.arrowedLine(
                    frame, (tcx, tcy), (bcx, bcy), (255, 0, 255), 2, tipLength=0.04
                )

        dashboard = render_black_dashboard(tracked_blue, tennis_ball)

        cv2.imshow("Live Camera Feed - Multi-Blue & Tennis Ball Tracker", frame)
        cv2.imshow("Blue Objects Telemetry (Pitch-Black Window)", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            blue_tracker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
