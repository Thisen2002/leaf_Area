import cv2
import cv2.aruco as aruco
import numpy as np

# --- CONFIGURATION ---
# The real-world dimensions of the rectangle defined by the markers (in cm)
# For example, if you place markers on the corners of an A4 paper:
# REAL_WIDTH_CM = 21.0
# REAL_HEIGHT_CM = 29.7
REAL_WIDTH_CM = 17.5
REAL_HEIGHT_CM = 25.5
SCALE = 100  # How many pixels per cm in the warped image (higher = better quality, slower)

def main():
    # 1. Open the Webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Load the ArUco dictionary (We use 4x4 markers)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Detect ArUco Markers
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        
        # Draw markers on the original frame for visual feedback
        aruco.drawDetectedMarkers(frame, corners, ids)

        # We need all 4 specific markers (0, 1, 2, 3) to proceed
        if ids is not None and len(ids) >= 4:
            # Flatten the ids list for easier searching
            ids = ids.flatten()
            
            # Check if we have the specific corner IDs we need
            # We use a set for O(1) lookup
            detected_ids = set(ids)
            required_ids = {0, 1, 2, 3}
            
            if required_ids.issubset(detected_ids):
                # 3. Organize the corner points
                # We need to map: ID 0 -> TopLeft, ID 1 -> TopRight, etc.
                src_pts = []
                
                # Get the center point of each marker to be the anchor
                for target_id in [0, 1, 2, 3]:
                    index = np.where(ids == target_id)[0][0]
                    # corners[index] is a list of 4 points for that marker. 
                    # We take the center of the marker (mean of the 4 corners)
                    marker_center = np.mean(corners[index][0], axis=0)
                    src_pts.append(marker_center)

                src_pts = np.array(src_pts, dtype="float32")

                # 4. Define Destination Points (The "Perfect" Flat View)
                # 0: TL, 1: TR, 2: BR, 3: BL
                dst_pts = np.array([
                    [0, 0],
                    [REAL_WIDTH_CM * SCALE, 0],
                    [REAL_WIDTH_CM * SCALE, REAL_HEIGHT_CM * SCALE],
                    [0, REAL_HEIGHT_CM * SCALE]
                ], dtype="float32")

                # 5. Warp Perspective
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(frame, matrix, (int(REAL_WIDTH_CM * SCALE), int(REAL_HEIGHT_CM * SCALE)))

                # 6. Segment the Leaf
                # Convert to Grayscale
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                # Blur to remove noise
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                # Otsu's Thresholding (Auto-finds best separation between leaf and paper)
                _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Optional: Morphological operations to remove small noise dots
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

                # 7. Calculate Area
                pixel_count = cv2.countNonZero(mask)
                # Area = (Total Pixels) * (Area of 1 pixel)
                # Since SCALE = 100 px/cm, 1 pixel is (1/100) cm wide.
                # Area of 1 pixel is (1/100) * (1/100) cm^2
                area_cm2 = pixel_count * (1 / (SCALE * SCALE))

                # --- VISUALIZATION ---
                # Show the warped view and the calculated area
                # Calculate roughly the center
                h, w = warped.shape[:2]
                center_x = int(w / 2) - 100 # -100 to shift text left so it looks centered
                center_y = int(h / 2) 
                
                cv2.putText(warped, f"Area: {area_cm2:.2f} cm2", (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
                
                # Show the mask (Black/White view of what the computer "sees" as the leaf)
                # Resize mask for display
                mask_display = cv2.resize(mask, (300, 400))
                warped_display = cv2.resize(warped, (300, 400))
                
                cv2.imshow("Warped View", warped_display)
                cv2.imshow("Leaf Mask", mask_display)

        cv2.imshow("Original Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()