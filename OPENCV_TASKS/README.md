# Blue Object Selection and Tennis Ball Detection using OpenCV

A real-time computer vision system written in Python using OpenCV and NumPy. This application captures a live camera stream to simultaneously detect a **tennis ball** and **select/track blue objects** of arbitrary geometry, computing real-time spatial metrics and relative position vectors between the two targets.

---

## 1. Approach to Deal with the Problem

To reliably detect and track two distinct objects with different geometric assumptions in real time, the problem was broken down into four key design requirements:

1. **Decoupled Detection Pipelines:** Because the blue object and tennis ball have distinct chromatic and geometric properties, they are processed through separate binary masks derived from a shared HSV frame.
2. **Illumination-Resilient Color Space:** Standard RGB/BGR channels are highly sensitive to shadows and ambient lighting changes. Converting frames to the **HSV (Hue, Saturation, Value)** color space isolates chromaticity (Hue) from brightness (Value).
3. **Shape-Agnostic vs. Shape-Constrained Geometry:**
   - **Blue Object:** Since no circular assumption can be made, the system relies on topological contour analysis (`cv2.contourArea`), spatial image moments (`cv2.moments`) to find the true center of mass, and a rectangular bounding box (`cv2.boundingRect`).
   - **Tennis Ball:** Since a tennis ball is spherical (projecting as a circle on the image plane), color segmentation is paired with a **circularity filter** to reject non-circular yellow-green background clutter before fitting a minimum enclosing circle (`cv2.minEnclosingCircle`).
4. **Interactive Selection & Temporal Tracking:** Users can either let the system automatically lock onto the primary blue object or left-click a specific blue object in the camera feed to track it across consecutive frames using nearest-centroid association and a `deque` trajectory buffer.

---

## 2. Method Used to Solve the Problem Statement

The frame-by-frame processing pipeline follows six sequential stages:

### Stage 1: Frame Acquisition & Preprocessing
- Frames are captured via `cv2.VideoCapture(0)` and horizontally flipped for intuitive visual feedback.
- A $7 \times 7$ **Gaussian Blur** (`cv2.GaussianBlur`) suppresses high-frequency sensor noise before color conversion via `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)`.

### Stage 2: HSV Thresholding & Morphological Filtering
- Two separate binary masks are generated using `cv2.inRange()`:
  - **Blue Mask:** Hue $\in [95, 135]$, Saturation $\in [100, 255]$, Value $\in [60, 255]$
  - **Tennis Ball Mask:** Hue $\in [24, 45]$, Saturation $\in [90, 255]$, Value $\in [80, 255]$
- Each mask undergoes **Morphological Opening** (erosion followed by dilation) to eliminate small background noise, followed by **Morphological Closing** (dilation followed by erosion) using an elliptical structuring element to fill small dark gaps on the object surface.

### Stage 3: Blue Object Identification & Tracking
- External contours are extracted via `cv2.findContours()`.
- **Area:** Calculated directly via Green's theorem using `cv2.contourArea(cnt)` in $\text{px}^2$, which accurately measures irregular and non-circular shapes.
- **Centroid:** Computed from zeroth and first-order spatial moments:
  $$c_x = \frac{M_{10}}{M_{00}}, \quad c_y = \frac{M_{01}}{M_{00}}$$
- **Bounding Box & Trajectory:** An upright bounding box $(x, y, w, h)$ is drawn around the contour, and the centroid coordinates are pushed to a fixed-length `collections.deque` to render a continuous tracking trail. If a user clicks on a specific blue object, nearest-neighbor Euclidean matching locks onto that target across frames.

### Stage 4: Tennis Ball Detection
- Candidate contours in the yellow-green mask are filtered by area and **isoperimetric circularity**:
  $$\text{Circularity} = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$
- Contours exceeding the circularity threshold ($0.65$) are enclosed using `cv2.minEnclosingCircle(cnt)`, yielding the sub-pixel **Center** $(t_x, t_y)$, **Radius** $r$ (in $\text{px}$), and **Area** ($\pi r^2$ in $\text{px}^2$).

### Stage 5: Relative Spatial Vector Computation
- When both the blue object and the tennis ball are present in the frame, the relative position of the blue object with respect to the tennis ball is computed as:
  $$\Delta x = c_x - t_x, \quad \Delta y = c_y - t_y, \quad d = \sqrt{\Delta x^2 + \Delta y^2}$$
- A directed vector (`cv2.arrowedLine`) is drawn from the tennis ball's center to the blue object's center alongside the live $(\Delta x, \Delta y)$ offset and Euclidean distance in pixels.

---

## 3. Other Possible Methods

While HSV thresholding + contour geometry provides high frame rates on CPU without external weights, several alternative approaches can solve this task depending on environmental constraints:

| Method | How It Works | Advantages | Limitations |
| :--- | :--- | :--- | :--- |
| **1. Hough Circle Transform (`cv2.HoughCircles`)** | Uses Canny edge detection and an accumulator voting space to detect circular shapes for the tennis ball. | Detects spheres even under partial color washout. | Computationally sensitive to edge noise; often triggers false positives on textured backgrounds; cannot handle non-circular blue objects. |
| **2. OpenCV Correlation Trackers (CSRT / KCF)** | User draws a bounding box via `cv2.selectROI()`; the tracker uses Discriminative Correlation Filters (`TrackerCSRT`) or Kernelized Correlation Filters (`TrackerKCF`). | Tracks arbitrary objects regardless of color Lighting shifts; great for manual one-time selection. | Struggles to recover if the object leaves the frame and re-enters; bounding box does not provide exact pixel area of irregular shapes without segmentation. |
| **3. Kalman Filter + Hungarian Algorithm** | Combines color/contour detection with a linear quadratic estimation (Kalman Filter) state model $[x, y, v_x, v_y]^T$. | Predicts object position during brief full occlusions and smooths jittery trajectories. | Adds state-tuning complexity unnecessary for single-camera unoccluded tracking. |
| **4. Deep Learning (YOLOv8 / YOLO11 + ByteTrack)** | Uses a pretrained convolutional neural network (`sports ball` class in COCO for the tennis ball, or custom segmentation model). | Highly robust to extreme lighting shifts, reflections, and complex backgrounds. | Requires higher compute (GPU recommended for high FPS) and custom training/segmentation masks to extract arbitrary blue objects' exact pixel areas. |
| **5. Lucas-Kanade Sparse Optical Flow (`cv2.calcOpticalFlowPyrLK`)** | Tracks Shi-Tomasi corner features inside the selected blue object across consecutive grayscale frames. | Fast motion tracking independent of color calibration after initial selection. | Suffers from drift over long sequences and fails on textureless surfaces. |

---

