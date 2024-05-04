""" HW10 - Object detection and tracking using image keypoints and descriptors.
    
    This script use SIFT keypoints and descriptors to develop an object
    detection and tracking application using Python programming.

    Authors: Jorge Rodrigo Gómez Mayo 
    Contact: jorger.gomez@udem.edu
    Organization: Universidad de Monterrey
    First created on Monday 29 April 2024

    Usage Examples:
        Only with requiered arguments:
            py hw10.py -obj .\obj_ref\bic.jpg -vid .\scenes\scene_2.mp4
        
        With all arguments:
            py hw10.py -obj .\obj_ref\bic.jpg -vid .\scenes\scene_2.mp4 -scale 15 -demo True
"""
# Import std libraries
import argparse
import copy
import cv2
import numpy as np

def parse_user_data() -> tuple[str, str]:
    """
    Parse the command-line arguments provided by the user.

    Returns:
        tuple[str, str]: A tuple containing the path to the object image and the input image.
    """
    parser = argparse.ArgumentParser(prog='CV - HW10',
                                    description="Object detection and tracking application based on SIFT keypoints and descriptors.", 
                                    epilog=' JRGM - 2024')
    parser.add_argument('-obj',
                        '--object_image',
                        type=str,
                        required=True,
                        help="Path to the image containing the object to track")
    parser.add_argument('-vid',
                        '--scene_video',
                        type=str,
                        required=True,
                        help="Path to the vide where we want to detect the object")
    parser.add_argument('-scale',
                        '--scale_percentage',
                        type=float,
                        required=False,
                        help="A value between 0 and 100 to indicate the rezise factor",
                        default=15)
    parser.add_argument('-demo',
                        '--demo_mode',
                        type=str,
                        required=False,
                        help="Enable/Disable Demo mode",
                        default="False")
    
    args = parser.parse_args()
    return args

def resize_image(img: np.ndarray, scale=100) -> np.ndarray:
    """
    Resize the image to a specified scale for better visualization.

    Args:
        img (np.ndarray): Image to resize.

    Returns:
        np.ndarray: Resized image.
    """
    width = int(img.shape[1] * (scale/100))
    height = int(img.shape[0] * (scale/100))
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def visualise_image(img: np.ndarray, title: str, scale=100, picture=True) -> None:
    """
    Display the image in a window with a title.

    Args:
        img (np.ndarray): Image to display.
        title (str): Title of the window.

    Returns:
        None
    """
    resized = resize_image(img, scale)
    cv2.imshow(title, resized)
    if picture:
        cv2.waitKey(0)

def close_windows(cap):
    """
    Close & destroy OpenCV windows

    The Function closes and destroy all cv2 windows that are open.
    """
    cap.release()
    cv2.destroyAllWindows()

def extract_features(img: np.ndarray, color_space="gray") -> tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT features from the image.

    Args:
        img (np.ndarray): Image data in which to find keypoints.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the keypoints and descriptors for the image.
    """
    match color_space:
        case "gray": 
            # Convert image to grayscale
            color_space = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        case "hsv":
            color_space = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(color_space, None)
    return keypoints, descriptors

def draw_keypoints(img: np.ndarray, color_space="gray", kp_type="rich") -> np.ndarray:
    """
    Draw keypoints on the image.

    Args:
        img (np.ndarray): Image on which to draw keypoints.

    Returns:
        np.ndarray: Image with keypoints drawn.
    """
    if "frame" in img:
        match color_space:
            case "gray": 
                # Convert image to grayscale
                color_space = cv2.cvtColor(img["frame"], cv2.COLOR_BGR2GRAY)

            case "hsv":
                color_space = cv2.cvtColor(img["frame"], cv2.COLOR_BGR2HSV)
            
            case "bgr":
                color_space = img["frame"]
    
    else:
        match color_space:
            case "gray": 
                # Convert image to grayscale
                color_space = cv2.cvtColor(img["path"], cv2.COLOR_BGR2GRAY)

            case "hsv":
                color_space = cv2.cvtColor(img["path"], cv2.COLOR_BGR2HSV)
            
            case "bgr":
                color_space = img["path"]

    match kp_type:
        case "rich":
            img_with_kp = cv2.drawKeypoints(color_space, img["features"]["kp"], None, color=(0,255,0),
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        case _:
            img_with_kp = cv2.drawKeypoints(color_space, img["features"]["kp"], None, color=(255, 0, 0))

    return img_with_kp

def match_features(descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Match SIFT features using the FLANN based matcher.

    Args:
        descriptors_1 (np.ndarray): Descriptors from the first image.
        descriptors_2 (np.ndarray): Descriptors from the second image.

    Returns:
        tuple[np.ndarray, list]: A tuple containing the matches and the mask for good matches.
    """
    # Define FLANN-based matcher parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Use knnMatch to find matches
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test
    matches_mask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.70 * n.distance:
            matches_mask[i] = [1, 0]
    return matches, matches_mask

def draw_matches(img_1: np.ndarray, img_2: np.ndarray, matches: np.ndarray, mask: list) -> np.ndarray:
    """
    Draw matches between two images.

    Args:
        img_1 (np.ndarray): First image.
        img_2 (np.ndarray): Second image.
        matches (np.ndarray): Matched features.
        mask (list): Mask for good matches.

    Returns:
        np.ndarray: Image with matches drawn.
    """
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=mask, flags=cv2.DrawMatchesFlags_DEFAULT)
    img = cv2.drawMatchesKnn(img_1["path"], img_1["features"]["kp"], img_2["frame"], img_2["features"]["kp"], matches, None, **draw_params)
    return img

def draw_rectangle(img, centroid, width=100, height=100, color=(0,0,255)):
    """
    Draw a straight bounding rectangle around a set of points.

    Args:
        frame (np.ndarray): The image on which to draw the rectangle.
        points (list of tuples): The points around which to draw the bounding rectangle.

    Returns:
        np.ndarray: The image with the bounding rectangle drawn.
    """
    if centroid:
        centroid_x, centroid_y = centroid
        # Calculate top-left corner of the rectangle
        top_left_x = int(centroid_x - width / 2)
        top_left_y = int(centroid_y - height / 2)
        # Draw the rectangle
        cv2.rectangle(img, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), color, 2)
    return img

def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32) # Adjust this for more/less prediction noise
    kalman.measurementNoiseCov = 1e-4 * np.array([[1, 0], [0, 1]], np.float32) # Adjust this for more/less measurement noise
    kalman.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
    return kalman

def update_kalman(kalman, measurement):
    predicted = kalman.predict()
    corrected = kalman.correct(measurement)
    corrected_x = int(corrected[0][0])
    corrected_y = int(corrected[1][0])
    predicted_x = int(predicted[0][0])
    predicted_y = int(predicted[1][0])
    return corrected_x, corrected_y, predicted_x, predicted_y

def run_pipeline():
    """
    Execute the object detection and tracking pipeline.

    Returns:
        None
    """
    # Create dictionaries to contain data
    print("Initializing...", end="\r")
    obj =   {"path": "", 
            "features": {"kp": "", "descriptors": ""}}
    scene = {"path": "",
            "features": {"kp": "", "descriptors": ""},
            "frame": ""}
    run =   {"scale": "",
            "mode": ""}
    
    # "Running" symbols list
    symbols = ["\u22ee","\u22f0","\u22ef","\u22f1"]
    
    # Counters, buffers and accumulators
    i_max = len(symbols)
    i = 0
    frames_since_last_detection = 0
    max_frames_without_detection = 5
    val_l_2_r = 0                       # Counter for left to right crossings
    val_r_2_l = 0                       # Counter for right to left crossing
    centroid_x_buffer = []              # Buffer to store the last few centroid's x positions
    buffer_size = 3                     # Size of the centroid buffer
    
    # Parse user's input data
    user_input = parse_user_data()
    run["scale"] = user_input.scale_percentage
    run["mode"] = str.lower(user_input.demo_mode)

    # Initialize Kalman Filter
    print("                                                  ", end="\r")
    print("Initializing Kalman Filter...", end="\r")
    kalman = initialize_kalman()
    
    # Load images
    print("                                                  ", end="\r")
    print("Loading object image...", end="\r")
    obj["path"] = cv2.imread(user_input.object_image)

    # Extract img features
    print("                                                  ", end="\r")
    print("Extracting object features...", end="\r")
    obj["features"]["kp"], obj["features"]["descriptors"] = extract_features(obj["path"], "hsv")

    # Draw Keypoints
    if run["mode"] == "true":
        print("                                                  ", end="\r")
        print("Drawing keypoints...", end="\r")
        obj_img_with_kp = draw_keypoints(obj)

        # Display image with keypoints
        visualise_image(obj_img_with_kp, "Object's Keypoints", run["scale"])

    # Define the HSV color range for segmentation
    lower_hsv1 = np.array([40, 50, 100])  
    upper_hsv1 = np.array([80, 255, 255])
    lower_hsv2 = np.array([20, 100, 100])
    upper_hsv2 = np.array([40, 255, 255])
    lower_hsv3 = np.array([0, 100, 100])
    upper_hsv3 = np.array([10, 255, 255])
    lower_hsv4 = np.array([160, 100, 100])
    upper_hsv4 = np.array([180, 255, 255])

    # Load video
    print("                                                  ", end="\r")
    print("Loading video...", end="\r")
    scene["path"] = cv2.VideoCapture(user_input.scene_video)
    
    # Initialize video frame loop
    print("                                                  ", end="\r")
    while True:
        i = i % i_max
        print("Tracking {0}".format(symbols[i]), end="\r")
        ret, scene["frame"] = scene["path"].read()
        if not ret:
            break

        # Create an HSV copy of the frame
        hsv_frame = cv2.cvtColor(copy.deepcopy(scene["frame"]), cv2.COLOR_BGR2HSV)

        # Create a copy of the frame where all the additions will be made
        tracking_frame = copy.deepcopy(scene["frame"])

        # Get frame dimensions
        height, width = tracking_frame.shape[:2]

        # Calculate the central vertical line
        center_x = width // 2

        # Draw a vertical line down the center of the frame
        cv2.line(tracking_frame, (center_x, 0), (center_x, height), (0, 255, 255), 3) 

        
        # Create masks for the colors
        color_mask1 = cv2.inRange(hsv_frame, lower_hsv1, upper_hsv1)
        color_mask2 = cv2.inRange(hsv_frame, lower_hsv2, upper_hsv2)
        color_mask3 = cv2.inRange(hsv_frame, lower_hsv3, upper_hsv3)
        color_mask4 = cv2.inRange(hsv_frame, lower_hsv4, upper_hsv4)

        # Combine the masks
        combined_color_mask = cv2.bitwise_or(color_mask1, color_mask2)
        combined_color_mask = cv2.bitwise_or(combined_color_mask, color_mask3)
        combined_color_mask = cv2.bitwise_or(combined_color_mask, color_mask4)
        
        # Show color mask
        if run["mode"] == "true":
            cv2.imshow("Mask",combined_color_mask)

        # Extract img features
        scene["features"]["kp"], scene["features"]["descriptors"] = extract_features(scene["frame"], "hsv")

        # Match features
        matches, matches_mask = match_features(obj["features"]["descriptors"], scene["features"]["descriptors"])

        # Draw matches
        if run["mode"] == "true":
            img_with_matches = draw_matches(obj, scene, matches, matches_mask)

            # Show Matches
            visualise_image(img_with_matches, "Matches", 25, False)

        # Calculate centroid and add matching key points (not addition as in 1+1=2)
        if matches:
            # Filter matches with masks
            good_points = [scene["features"]["kp"][matches[i][0].trainIdx].pt for i in range(len(matches)) if matches_mask[i][0] 
                                            and combined_color_mask[int(scene["features"]["kp"][matches[i][0].trainIdx].pt[1]), 
                                            int(scene["features"]["kp"][matches[i][0].trainIdx].pt[0])] != 0]
            if good_points:
                frames_since_last_detection = 0  # Reset the counter
                centroid_x = int(sum(x for x, y in good_points) / len(good_points))
                centroid_y = int(sum(y for x, y in good_points) / len(good_points))
                
                measurement = np.array([[np.float32(centroid_x)], [np.float32(centroid_y)]])
                x, y, x2, y2 = update_kalman(kalman, measurement)

                # Draw lines between real, predicted, and corrected positions
                if run["mode"] == "true":
                    cv2.line(tracking_frame, (centroid_x, centroid_y), (x2, y2), (0, 0, 0), 2)  # Black line: Real to Predicted
                    cv2.line(tracking_frame, (centroid_x, centroid_y), (x, y), (255, 255, 255), 2)  # White line: Real to Corrected

                # Add matching kp
                for point in good_points:
                    cv2.circle(tracking_frame, (int(point[0]), int(point[1])), 4, (255, 255, 0), -1)

            else:
                # Using kalman filter (Purple box)
                if run["mode"] == "true":
                    # Add object's centroid 
                    cv2.circle(tracking_frame, (x, y), 5, (255, 0, 255), -1)

                    # Add object's bounding rectangle
                    tracking_frame = draw_rectangle(tracking_frame, (x, y), 70, 170, color=(255,0,255))
                x, y = (centroid_x, centroid_y) # Usiing "real" value before kalman filter
                print("No detection. Using last centroid position before Kalman filter.")
                if frames_since_last_detection > max_frames_without_detection:
                    print("Object lost. Stopping tracking.")
                    break  

                if x is None and y is None:
                    print("No previous known position. Object not yet detected.")
                    continue
                else:
                    frames_since_last_detection += 1

        if x is not None and y is not None:
            # Add object's centroid
            cv2.circle(tracking_frame, (x, y), 5, (0, 0, 255), -1)

            # Add object's bounding rectangle
            tracking_frame = draw_rectangle(tracking_frame, (x, y), 70, 170)

            # Update centroid buffer
            centroid_x_buffer.append(x)
            if len(centroid_x_buffer) > buffer_size:
                centroid_x_buffer.pop(0)

        # Chek for direction changes
        if len(centroid_x_buffer) >= 2:
            old_x = centroid_x_buffer[-2]
            new_x = centroid_x_buffer[-1]

            # Check crossing from left to right
            if old_x < center_x and new_x >= center_x:
                val_l_2_r += 1
        
            # Check crossing from right to left
            if old_x > center_x and new_x <= center_x:
                val_r_2_l += 1

        # Visualize important data on the frame
        text_title = "Visual analytics"
        text_l_2_r = f"Object passing the reference line from left to right: {val_l_2_r}"
        text_r_2_l = f"Object passing the reference line from right to left: {val_r_2_l}"
        text_total = f"Total crossings of the reference line in either direction: {val_l_2_r+val_r_2_l }"
        cv2.putText(tracking_frame, text_title, (15, height - 92), cv2.FONT_HERSHEY_DUPLEX, 0.75, (128, 0, 0), 1)
        cv2.putText(tracking_frame, text_l_2_r, (15, height - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 0, 0), 1)
        cv2.putText(tracking_frame, text_r_2_l, (15, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 0, 0), 1)
        cv2.putText(tracking_frame, text_total, (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 0, 0), 1)


        # Show Results
        visualise_image(tracking_frame, "Object Tracking", picture=False)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    print("Execution Summary:\n", end="\r")
    print("Object passing the reference line from left to right: {}".format(val_l_2_r),
            "Object passing the reference line from right to left: {}\n".format(val_r_2_l), 
            end="\r", sep="\n")
    print("Shutting down...", end="\r")
    close_windows(scene["path"])


if __name__ == "__main__":
    run_pipeline()

"""
    References:
"""