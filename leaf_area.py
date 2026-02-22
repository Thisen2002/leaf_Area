import cv2
import cv2.aruco as aruco
import numpy as np

def calculate_leaf_area_with_aruco(image, real_width_cm, real_height_cm):
    # 1. Initialize ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    
    # 2. Detect Markers
    corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    
    # We need 4 specific markers (IDs 0, 1, 2, 3) to define the board
    if ids is not None and len(ids) >= 4:
        # Sort markers by ID so we know which is Top-Left, Top-Right, etc.
        # This ensures the board is always oriented correctly, even if the phone is upside down
        ids = ids.flatten()
        
        # Create a dictionary to map ID -> Corner Coordinate
        # We use the top-left corner of each marker as the reference point
        centers = {}
        for i, corner in zip(ids, corners):
            centers[i] = corner[0][0] # Taking the first corner of the marker

        # Check if we found IDs 0, 1, 2, 3
        if all(id in centers for id in [0, 1, 2, 3]):
            src_pts = np.array([centers[0], centers[1], centers[2], centers[3]], dtype="float32")
            
            # Destination points (The "Perfect" flat view)
            # 0: TL, 1: TR, 2: BR, 3: BL
            scale = 100 # pixels per cm
            dst_pts = np.array([
                [0, 0],
                [real_width_cm * scale, 0],
                [real_width_cm * scale, real_height_cm * scale],
                [0, real_height_cm * scale]
            ], dtype="float32")

            # 3. Warp Perspective (The "Un-squash" step)
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, matrix, (int(real_width_cm*scale), int(real_height_cm*scale)))

            # 4. Segment Leaf (Green color extraction or Thresholding)
            # Convert to HSV is usually better for separating green leaves from white paper
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            
            # Define range for green color (This might need tweaking based on lighting)
            # Or use the Thresholding method from previous example if paper is white
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 5. Calculate Area
            pixel_count = cv2.countNonZero(mask)
            area_cm2 = pixel_count * (1 / (scale * scale))
            
            return area_cm2, warped

    return None, None

# Usage note:
# You would feed the camera frame into this function.
# If it returns an area, you show it on the screen.


# Example usage with a test image
image = cv2.imread('/home/thisen/Desktop/new leaf/Qr creating/22.png')
real_width_cm = 20
real_height_cm = 30

area, warped_image = calculate_leaf_area_with_aruco(image, real_width_cm, real_height_cm)

if area is not None:
    print(f"Estimated Leaf Area: {area:.2f} cm^2")
    #cv2.imshow("Warped Image", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not detect all required markers.")