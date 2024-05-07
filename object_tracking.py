""" HW10 - Object detection and tracking using image keypoints and descriptors.
    
    This script use SIFT keypoints and descriptors to develop an object
    detection and tracking application using Python programming.

    Authors: Jorge Rodrigo Gómez Mayo 
    Contact: jorger.gomez@udem.edu
    Organization: Universidad de Monterrey
    First created on Monday 29 April 2024

    Usage Examples:
        Only with requiered arguments:
            py object_tracking.py -obj .\obj_ref\bic.jpg -vid .\scenes\scene_1.mp4
        
        With all arguments:
            py object_tracking.py -obj .\obj_ref\bic.jpg -vid .\scenes\scene_1.mp4 -scale 15 -demo True
"""
# Import std libraries
import argparse
import copy
import cv2
import numpy as np

def parse_user_data() -> tuple[str, str, float, str]:
    """
    Parse command-line arguments to get the paths for the object image and video, as well as other parameters for the application.

    Returns:
        tuple[str, str, float, str]: A tuple containing the path to the object image, the video file, 
        the scale percentage, and the demo mode status.
    """
    # Setup argument parser with descriptions
    parser = argparse.ArgumentParser(prog='CV - HW10',
                                    description="Object detection and tracking application based on SIFT keypoints and descriptors.", 
                                    epilog=' JRGM - 2024')
    
    # Define the expected command-line arguments
    parser.add_argument('-obj',
                        '--object_image',
                        type=str,
                        required=True,
                        help="Path to the image containing the object to track")
    parser.add_argument('-vid',
                        '--scene_video',
                        type=str,
                        required=True,
                        help="Path to the video where we want to detect the object")
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

def resize_image_to_height(img: np.ndarray, target_height: int) -> np.ndarray:
    """
    Resize the image to match a specified target height while preserving the aspect ratio.

    Args:
        img (np.ndarray): The image to be resized.
        target_height (int): The target height in pixels.

    Returns:
        np.ndarray: The resized image.
    """
    current_height = img.shape[0]
    scale = target_height / current_height
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

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

def visualise_image(img: np.ndarray, title: str, scale: int = 100, picture: bool = True) -> None:
    """
    Display an image in a window with a specified title and scale.

    Args:
        img (np.ndarray): The image to display.
        title (str): The title of the window.
        scale (int): Percentage to scale the image for display.
        picture (bool): Flag to wait for a key press if true.

    Returns:
        None
    """
    resized = resize_image(img, scale)
    cv2.imshow(title, resized)
    if picture:
        cv2.waitKey(0)
    return None

def close_windows(cap) -> None:
    """
    Close & destroy OpenCV windows

    The Function closes and destroy all cv2 windows that are open.
    """
    cap.release()
    cv2.destroyAllWindows()
    return None

def extract_features(img: np.ndarray, color_space: str = "gray") -> tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT features from the image using a specified color space.

    Args:
        img (np.ndarray): The image from which to extract features.
        color_space (str): Color space conversion before feature extraction ('gray', 'hsv', 'bgr').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of keypoints and descriptors.
    """
    # Convert the image to the specified color space
    match color_space:
        case "gray": 
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
    # Check if it is a frame or a picture, then Convert to the specified color space
    if "frame" in img:
        match color_space:
            case "gray": 
                color_space = cv2.cvtColor(img["frame"], cv2.COLOR_BGR2GRAY)

            case "hsv":
                color_space = cv2.cvtColor(img["frame"], cv2.COLOR_BGR2HSV)
            
            case "bgr":
                color_space = img["frame"]
    
    else:
        match color_space:
            case "gray": 
                color_space = cv2.cvtColor(img["path"], cv2.COLOR_BGR2GRAY)

            case "hsv":
                color_space = cv2.cvtColor(img["path"], cv2.COLOR_BGR2HSV)
            
            case "bgr":
                color_space = img["path"]

    # Choose kp type to be displayed
    match kp_type:
        case "rich":
            img_with_kp = cv2.drawKeypoints(color_space, img["features"]["kp"], None, color=(255,255,0),
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
    index_params = dict(algorithm=1, trees=350)
    search_params = dict(checks=1)

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

def draw_matches(img_1: np.ndarray, img_2: np.ndarray, matches: list[cv2.DMatch], mask: list[int]) -> np.ndarray:
    """
    Draw lines between matching keypoints in two images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        matches (list[cv2.DMatch]): List of matched keypoints.
        mask (list[int]): List indicating which matches are good.

    Returns:
        np.ndarray: The image with drawn matches.
    """
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=mask, flags=cv2.DrawMatchesFlags_DEFAULT)
    img = cv2.drawMatchesKnn(img_1["path"], img_1["features"]["kp"], img_2["frame"], img_2["features"]["kp"], matches, None, **draw_params)
    return img

def find_contours(mask):
    """
    Find contours in a given mask.

    Args:
        mask (np.ndarray): Binary mask where contours are to be found.

    Returns:
        list: List of contour points.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

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

def run_pipeline() -> None:
    """
    Execute the object detection and tracking pipeline.

    Returns:
        None
    """
    try:
        # Create dictionaries to contain data for object, scene, and runtime parameters
        print("Initializing...", end="\r")
        obj =   {"path": "", 
                "features": {"kp": "", "descriptors": ""}}
        scene = {"path": "",
                "features": {"kp": "", "descriptors": ""},
                "frame": ""}
        run =   {"scale": "",
                "mode": ""}

        # "Tracking" symbols list
        symbols = ["\u22ee","\u22f0","\u22ef","\u22f1"]

        # Initialize counters and buffers for tracking and direction change detection
        i_max = len(symbols)
        i = 0                               # Symbol index
        frames_since_last_detection = 0     
        max_frames_without_detection = 5    # Must be bigger than 0
        centroid_x_buffer = []              # Buffer to store the last few centroid's x positions
        buffer_size = 2                     # Size of the centroid buffer
        val_l_2_r = 0                       # Counter for left to right crossings
        val_r_2_l = 0                       # Counter for right to left crossing
        last_incremented = None             # Can be 'left_to_right' or 'right_to_left

        # Parse command-line arguments to get user-specified configurations
        user_input = parse_user_data()
        run["scale"] = user_input.scale_percentage
        run["mode"] = str.lower(user_input.demo_mode)
        print("                                                  ", end="\r")
        print("Setting up execution mode...", end="\r")
        print("                                                  ", end="\r")
        if run["mode"] == "true":
            print("Running in demo mode:\n", end="\r")
            print('\tTo stop execution press "Ctrl + c" in terminal and then any key with an output window open')
            print('\tTo avoid any errors, please wait until "Object Tracking" window is open before stoping')
        else:
            print("Running in normal mode:\n", end="\r")
            print("\tTo stop exectuion press q")

        # Load the object image and validate it
        print("                                                  ", end="\r")
        print("Loading object image...", end="\r")
        obj["path"] = cv2.imread(user_input.object_image)
        if obj["path"] is None:
            raise Exception("Failed to load the object image.")

        # Load the video and validate it
        print("                                                  ", end="\r")
        print("Loading video...", end="\r")
        scene["path"] = cv2.VideoCapture(user_input.scene_video)
        if not scene["path"].isOpened():
            raise Exception("Failed to open video file.")

        if run["mode"] == "true":
            # Adjust object image size to match video frame size for better comparison
            print("                                                  ", end="\r")
            print("Matching sizes...", end="\r")
            ret, scene["frame"] = scene["path"].read()
            if not ret:
                raise Exception("Failed to read the first frame from the video.")
            
            obj["path"] = resize_image_to_height(obj["path"], scene["frame"].shape[0])

        # Extract features from the object image using SIFT
        print("                                                  ", end="\r")
        print("Extracting object features...", end="\r")
        obj["features"]["kp"], obj["features"]["descriptors"] = extract_features(obj["path"], "hsv")

        # Optionally, draw keypoints on the object image for visualization in demo mode
        if run["mode"] == "true":
            print("                                                  ", end="\r")
            print("Drawing keypoints...", end="\r")
            obj_img_with_kp = draw_keypoints(obj, "hsv")
            visualise_image(obj_img_with_kp, "Object's Keypoints")

        # Define HSV color ranges for object segmentation by color
        # Green
        lower_hsv1 = np.array([40, 50, 50])
        upper_hsv1 = np.array([80, 255, 255])
        # Yellow
        lower_hsv2 = np.array([20, 100, 100])
        upper_hsv2 = np.array([40, 255, 255])
        # Reds
        lower_hsv3 = np.array([0, 100, 100])
        upper_hsv3 = np.array([10, 255, 255])
        lower_hsv4 = np.array([160, 100, 100])
        upper_hsv4 = np.array([180, 255, 255])
        # White
        lower_hsv5 = np.array([0, 0, 200])
        upper_hsv5 = np.array([180, 50, 255])

        # Initialize the main video processing loop
        print("                                                  ", end="\r")
        while True:
            i = i % i_max
            print("Tracking {0}".format(symbols[i]), end="\r")
            ret, scene["frame"] = scene["path"].read()
            if not ret:
                break

            # Convert the current video frame to HSV for color-based processing
            hsv_frame = cv2.cvtColor(copy.deepcopy(scene["frame"]), cv2.COLOR_BGR2HSV)

            # Create a copy of the frame where all the additions will be made
            tracking_frame = copy.deepcopy(scene["frame"])

            # Get frame dimensions
            height, width = tracking_frame.shape[:2]

            # Calculate the position of vertical reference line
            reference_x = (width // 2) + 50

            # Draw a vertical line down the center of the frame
            cv2.line(tracking_frame, (reference_x, 0), (reference_x, height), (0, 255, 255), 3) 

            # Generate masks for different color segments and find contours
            color_mask1 = cv2.inRange(hsv_frame, lower_hsv1, upper_hsv1) # Green
            color_mask2 = cv2.inRange(hsv_frame, lower_hsv2, upper_hsv2) # Yellow
            color_mask3 = cv2.inRange(hsv_frame, lower_hsv3, upper_hsv3) # Red1
            color_mask4 = cv2.inRange(hsv_frame, lower_hsv4, upper_hsv4) # Red2
            color_mask5 = cv2.inRange(hsv_frame, lower_hsv5, upper_hsv5) # White

            # Find contours of the green mask
            contours = find_contours(color_mask1)  

            # Combine the masks
            combined_color_mask = cv2.bitwise_or(color_mask1, color_mask2)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, color_mask3)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, color_mask4)
            combined_color_mask = cv2.bitwise_or(combined_color_mask, color_mask5)
            
            # Optionally, Show color mask and draw mask contour on the tracking frame for visualization in demo mode
            if run["mode"] == "true":
                cv2.imshow("Mask",combined_color_mask)
                cv2.drawContours(tracking_frame, contours, -1, (255, 0, 0), 2)
                

            # Extract features from the video frame and match with the object feature
            scene["features"]["kp"], scene["features"]["descriptors"] = extract_features(scene["frame"], "hsv")
            try:
                matches, matches_mask = match_features(obj["features"]["descriptors"], scene["features"]["descriptors"])
            except cv2.error as e:
                if run["mode"] == "true":
                    print(f"Error detected during matching!\n{e}", end="\r")
                    matches = [["", ""],["",""]]
                    matches_mask = [["", ""],["",""]]
                else:
                    print(f"Error detected during matching!\n", end="\r")
                    matches = [["", ""],["",""]]
                    matches_mask = [["", ""],["",""]]

            # Optionally draw and display matches in demo mode
            if run["mode"] == "true":
                try:
                    img_with_matches = draw_matches(obj, scene, matches, matches_mask)
                    visualise_image(img_with_matches, "Matches", 80, False)
                except cv2.error as e:
                    continue

            # Calculate centroid of matching points to track the object
            if matches:
                # Filter "good points"
                good_points = [scene["features"]["kp"][matches[i][0].trainIdx].pt for i in range(len(matches)) 
                                if matches_mask[i][0] 
                                and (((len(contours) > 0 
                                    and cv2.pointPolygonTest(contours[0], (int(scene["features"]["kp"][matches[i][0].trainIdx].pt[0]),
                                    int(scene["features"]["kp"][matches[i][0].trainIdx].pt[1])), True) >= -1)) 
                                or combined_color_mask[int(scene["features"]["kp"][matches[i][0].trainIdx].pt[1]), 
                                                int(scene["features"]["kp"][matches[i][0].trainIdx].pt[0])] != 0)]
                
                if good_points:
                    frames_since_last_detection = 0  # Reset the counter
                    centroid_x = int(sum(x for x, y in good_points) / len(good_points))
                    centroid_y = int(sum(y for x, y in good_points) / len(good_points))

                    # Add matching kp
                    for point in good_points:
                        cv2.circle(tracking_frame, (int(point[0]), int(point[1])), 3, (255, 255, 0), -1)

                    # Add object's centroid
                    cv2.circle(tracking_frame, (centroid_x, centroid_y), 4, (0, 0, 255), -1)

                    # Add object's bounding rectangle
                    tracking_frame = draw_rectangle(tracking_frame, (centroid_x, centroid_y), 70, 170)

                    # Update centroid buffer
                    centroid_x_buffer.append(centroid_x)
                    if len(centroid_x_buffer) > buffer_size:
                        centroid_x_buffer.pop(0)

                # What to do when object is not detected
                else:
                    frames_since_last_detection += 1 
                    msg = "Object lost!"
                    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0]
                    text_x = width - text_size[0] - 10  # 10 pixels margin on the right
                    cv2.putText(tracking_frame, msg, (text_x, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    if frames_since_last_detection > max_frames_without_detection:
                        print("Object lost for too long! Stopping tracking...")
                        break       

            # Check for direction changes based on the centroid's position relative to the reference line
            if len(centroid_x_buffer) >= 2:
                old_x = centroid_x_buffer[-2]
                new_x = centroid_x_buffer[-1]

                # Check crossing from left to right
                if old_x < reference_x and new_x >= reference_x:
                    if last_incremented != 'left_to_right':
                        val_l_2_r += 1
                        last_incremented = 'left_to_right'

                # Check crossing from right to left
                if old_x > reference_x and new_x <= reference_x:
                    if last_incremented != 'right_to_left':
                        val_r_2_l += 1
                        last_incremented = 'right_to_left'

            # Display tracking and crossing information on the frame
            texts = ["Visual analytics",
                    f"Object passing the reference line from left to right: {val_l_2_r}",
                    f"Object passing the reference line from right to left: {val_r_2_l}",
                    f"Total crossings of the reference line in either direction: {val_l_2_r + val_r_2_l}"]

            y_pos = height - 100  # Start 100 pixels above the bottom edge
            for text in texts:
                if text == "Visual analytics":
                    # Black edge
                    cv2.putText(tracking_frame, text, (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    # White text
                    cv2.putText(tracking_frame, text, (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    # Black edge
                    cv2.putText(tracking_frame, text, (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 0, 0), 4, cv2.LINE_AA)
                    # White text
                    cv2.putText(tracking_frame, text, (15, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 24  # Adjust y position for the next text

            # Display the tracking frame
            if run["mode"] == "true":
                visualise_image(tracking_frame, "Object Tracking", picture=True)
            else:
                visualise_image(tracking_frame, "Object Tracking", picture=False)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # waiting for a 'q' key press to quit in non-demo mode
                break
            i += 1

        # Print a summary of object crossings after the loop finishes
        print("Execution Summary:\n", end="\r")
        print("\tObject passing the reference line from left to right: {}".format(val_l_2_r),
                "\tObject passing the reference line from right to left: {}".format(val_r_2_l),
                "\tTotal crossings of the reference line in either direction: {}\n".format(val_l_2_r + val_r_2_l),
                end="\r", sep="\n")
        print("Shutting down...", end="\r")
        close_windows(scene["path"])

    # Handle a keyboard interrupt for graceful shutdown
    except KeyboardInterrupt:
        print("Execution Summary:\n", end="\r")
        print("\tObject passing the reference line from left to right: {}".format(val_l_2_r),
                "\tObject passing the reference line from right to left: {}".format(val_r_2_l),
                "\tTotal crossings of the reference line in either direction: {}\n".format(val_l_2_r + val_r_2_l),
                end="\r", sep="\n")
        print("Shutting down...", end="\r")
        close_windows(scene["path"])
    
    return None 

if __name__ == "__main__":
    run_pipeline()

"""
    References:
    
    [1]“print() and Standard Out.” Accessed: May 05, 2024. [Online]. 
        Available: https://cs.stanford.edu/people/nick/py/python-print.html

    [2]“OpenCV: Contour Features.” Accessed: May 05, 2024. [Online]. 
        Available: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    
    [3]“OpenCV: Point Polygon Test.” Accessed: May 05, 2024. [Online]. 
        Available: https://docs.opencv.org/4.x/dc/d48/tutorial_point_polygon_test.html
    
    [4]“OpenCV: Feature Matching with FLANN.” Accessed: May 05, 2024. [Online]. 
        Available: https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html
    
    [5]“OpenCV: Feature Matching.” Accessed: May 05, 2024. [Online]. 
        Available: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    
    [6]“OpenCV: Introduction to SIFT (Scale-Invariant Feature Transform).” Accessed: May 05, 2024. 
        [Online]. Available: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
"""