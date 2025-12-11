import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from image_processing.process_input import PreprocessInput
from models.unet_model import UNetModel
from image_processing.process_output import PostProcessOutput
from itertools import combinations
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean

import sys
import os

CALIBRATION_FACTOR = 1.08

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Load the model
unet_model = UNetModel((256, 256, 1), weight_path=resource_path(r"models/fl_unet.h5")).get_model()

class FemurLengthEstimation:
    def __init__(self, threshold=20):
        """
        Initialize the FemurLengthEstimation class.
        
        Parameters:
        threshold (int): Threshold for postprocessing the binary segmentation mask.
        """
        # Postprocessing
        self.postprocess = PostProcessOutput(threshold)

    def estimate_femur_length(self, im):
        """
        Estimate femur length from the input ultrasound image.
        
        Parameters:
            image (PIL.Image): Input ultrasound image.
            
        Returns:
            float: Estimated femur length in pixels.
        """
        # Preprocess image
        image = im.copy()
        image = image.convert("L")
        size = image.size
        preprocessed_image = PreprocessInput('hc').preprocess_image(image)

        # Predict segmentation mask
        pred_mask = unet_model.predict(preprocessed_image, verbose=0)
        pred_mask = (pred_mask > 0.1).astype(np.uint8).reshape(256, 256)

        # Postprocess mask
        binary_mask = self.postprocess.process_binary(pred_mask)
        resized_mask = cv2.resize(binary_mask, size, interpolation=cv2.INTER_NEAREST)
        
        # skeletonize the mask
        skeleton = skeletonize(resized_mask)
        
        contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 2. Find the largest contour by length
        if not contours:
            femur_length = 0.0
            max_pair = None  # No contours found
            return None, None
        else:
            largest_contour = max(contours, key=lambda c: cv2.arcLength(c, closed=True))  # or len for perimeter

            # 3. Reshape to (N, 2)
            points = largest_contour[:, 0, :]  # shape: (N, 2)

            # 4. Compute the maximum Euclidean distance between all point pairs
            max_dist = 0.0
            for pt1, pt2 in combinations(points, 2):
                dist = euclidean(pt1, pt2)
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (pt1, pt2)

            femur_length = max_dist * CALIBRATION_FACTOR

        return femur_length, self.visualize_femur_length(image, max_pair)
    
    def visualize_femur_length(self, image, max_pair):
        """
        Visualize the femur length estimation on the image.
        
        Parameters:
            image (PIL.Image): Input ultrasound image.
            max_pair (tuple): Pair of points representing the femur length.
            
        Returns:
            PIL.Image: Image with femur length visualization.
        """
        # Convert to numpy array
        
        img_array = np.array(image.copy())

        pt1, pt2 = max_pair
        plt.imshow(img_array, cmap='gray')
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', linestyle='--', linewidth=2)
        plt.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', marker='x')
        plt.gcf().set_size_inches(image.width / plt.gcf().dpi, image.height / plt.gcf().dpi)
        plt.axis('off')
        plt.tight_layout(pad=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return Image.open(buf)