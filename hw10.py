""" HW10 - Object detection and tracking using image keypoints and descriptors.
    
    This script use image keypoints and descriptors to develop an object
    detection and tracking application using Python programming.

    Authors: Jorge Rodrigo Gómez Mayo 
    Contact: jorger.gomez@udem.edu
    Organization: Universidad de Monterrey
    First created on Monday 29 April 2024

    Usage Example:
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
    parser = argparse.ArgumentParser(prog='HW10 ',
                                    description='Description', 
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
                        required=True,
                        help="A value between 0 and 100 to indicate the rezise factor")
    parser.add_argument('-demo',
                        '--demo_mode',
                        type=str,
                        required=True,
                        help="Enable/Disable Demo mode")
    
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
    
    # Parse user's input data
    user_input = parse_user_data()
    run["scale"] = user_input.scale_percentage
    run["mode"] = str.lower(user_input.demo_mode)
    
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
    symbols = ["\u22ee","\u22f0","\u22ef","\u22f1"]
    i_max = len(symbols)
    i = 0
    while True:
        ret, scene["frame"] = scene["path"].read()
        if not ret:
            break

        i = i % i_max
        print("Tracking {0}".format(symbols[i]), end="\r")
        # Create an HSV copy of the frame
        hsv_frame = cv2.cvtColor(copy.deepcopy(scene["frame"]), cv2.COLOR_BGR2HSV)

        # Create a copy of the frame where all the additions will be made
        tracking_frame = copy.deepcopy(scene["frame"])
        
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

        # Calculate centroid and add matching key points
        if matches:
            # Filter matches with masks
            good_points = [scene["features"]["kp"][matches[i][0].trainIdx].pt for i in range(len(matches)) if matches_mask[i][0] and combined_color_mask[int(scene["features"]["kp"][matches[i][0].trainIdx].pt[1]), int(scene["features"]["kp"][matches[i][0].trainIdx].pt[0])] != 0]
            if good_points:
                centroid_x = int(sum(x for x, y in good_points) / len(good_points))
                centroid_y = int(sum(y for x, y in good_points) / len(good_points))
                cv2.circle(tracking_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                for point in good_points:
                    cv2.circle(tracking_frame, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)

        # Show Results
        visualise_image(tracking_frame, "Traking", picture=False)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    close_windows(scene["path"])

if __name__ == "__main__":
    run_pipeline()

"""
    References:
"""