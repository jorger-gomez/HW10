""" HW10 - Object detection and tracking using image keypoints and descriptors.
    
    This script use image keypoints and descriptors to develop an object
    detection and tracking application using Python programming.

    Authors: Jorge Rodrigo Gómez Mayo 
    Contact: jorger.gomez@udem.edu
    Organization: Universidad de Monterrey
    First created on Monday 29 April 2024

    Usage Example:
        py 
"""
# Import std libraries
import argparse
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
    
    args = parser.parse_args()
    return args

def resize_image(img: np.ndarray) -> np.ndarray:
    """
    Resize the image to a specified scale for better visualization.

    Args:
        img (np.ndarray): Image to resize.

    Returns:
        np.ndarray: Resized image.
    """
    width = int(img.shape[1] * 0.35)
    height = int(img.shape[0] * 0.35)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def visualise_image(img: np.ndarray, title: str) -> None:
    """
    Display the image in a window with a title.

    Args:
        img (np.ndarray): Image to display.
        title (str): Title of the window.

    Returns:
        None
    """
    resized = resize_image(img)
    cv2.imshow(title, resized)
    cv2.waitKey(0)

def close_windows():
    """
    Close & destroy OpenCV windows

    The Function closes and destroy all cv2 windows that are open.
    """
    cv2.destroyAllWindows()

def run_pipeline():
    """
    --

    Returns:
        None
    """

if __name__ == "__main__":
    run_pipeline()

"""
    References:
"""