import cv2
import cv2.aruco as aruco
import numpy as np

# --- CONFIGURATION ---
# Your measured real-world dimensions between marker centers (in cm)
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
            detected_ids = set(ids)
            required_ids = {0, 1, 2, 3}
            
            if required_ids.issubset(detected_ids):
                # 3. Organize the corner points
                src_pts = []
                
                # Get the center point of each marker to be the anchor
                for target_id in [0, 1, 2, 3]:
                    index = np.where(ids == target_id)[0][0]
                    marker_center = np.mean(corners[index][0], axis=0)
                    src_pts.append(marker_center)

                src_pts = np.array(src_pts, dtype="float32")

                # 4. Define Destination Points (The "Perfect" Flat View)
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
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Morphological operations to remove small noise dots
                kernel = np.ones((3,3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

                # --- NEW FIX: ISOLATE ONLY THE LEAF ---
                # Find all distinct shapes (contours) in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the contour with the largest area (ignores the 4 corner markers)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create a brand new, completely black mask
                    clean_mask = np.zeros_like(mask)
                    
                    # Fill ONLY the largest shape with white
                    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    
                    # 7. Calculate Area (Using our new clean_mask!)
                    pixel_count = cv2.countNonZero(clean_mask)
                    area_cm2 = pixel_count * (1 / (SCALE * SCALE))
                    
                    # Bonus: Draw a green outline around the detected object on the warped image
                    cv2.drawContours(warped, [largest_contour], -1, (0, 255, 0), 3)
                    
                else:
                    area_cm2 = 0.0
                    clean_mask = mask # Fallback if nothing is found

                # --- VISUALIZATION ---
                h, w = warped.shape[:2]
                center_x = int(w / 2) - 250 # Adjusted slightly to fit the large text
                center_y = int(h / 2) 
                
                cv2.putText(warped, f"Area: {area_cm2:.2f} cm2", (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5) # Scaled font down slightly from 5 to 3 to fit the screen better
                
                # Show the clean mask instead of the messy one
                mask_display = cv2.resize(clean_mask, (300, 400))
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