import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from image_processing.process_input import PreprocessInput
from models.xception_model import XceptionModel
from models.unet_model import UNetModel
from image_processing.process_output import PostProcessOutput
from utils.ellipse_fitting import EllipseFitting
import sys
import os
CALIBRATION_FACTOR=1.04

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
logo_path = resource_path("resources/logo.ico")

# Load the model
uspc_weights = resource_path(r"models/uspc_xception.h5")
bpc_weights = resource_path(r"models/bvc_xception.h5")
hc_weights = resource_path(r"models/hc_unet.h5")
uspc_model = XceptionModel((256,256,1), 6, weight_path=uspc_weights).get_model()
bpc_model = XceptionModel((256, 256, 3), 3, weight_path=bpc_weights).get_model()
hc_model = UNetModel((256, 256, 1), weight_path= hc_weights).get_model()

class BrainBiometryPipeline:
    def __init__(self, threshold = 50):
        """
        Initialize the BrainBiometryPipeline class.
        Parameters:
        Threshold: For postprocessing the binary segmentation mask

        """
        #postprocessing
        self.postprocess = PostProcessOutput(threshold)

        #ellipse fitting
        self.ellipse_fitting = EllipseFitting()

    def plane_classification(self, image):
        # USPC_classes = ['Abdomen', 'Femur', 'Cervix', 'Thorax', 'Brain', 'Other']
        USPC_classes = ['Abdomen', 'Brain', 'Femur', 'Thorax', 'Cervix', 'Other']
        """
        Identify the fetal plane.
        
        Parameters: 
            image (PIL.Image): Input ultrasound image.
            
        Returns:
         Identified Fetal Plane class.
        """

        # Preprocess image
        image = image.convert("L")
        image1 = PreprocessInput('uspc').preprocess_image(image)

        # Initial plane prediction
        pred1 = uspc_model.predict(image1, verbose=0)
        confidence = round(np.max(pred1[0]), 2)
        pred1 = (pred1 > 0.2).astype('uint8')
        pred_plane = USPC_classes[np.argmax(pred1, axis=1)[0]]


        return pred_plane, confidence
    
    def brain_view_classification(self, image):
        BPC_classes = ["Trans-cerebellar", "Trans-thalamic", "Trans-ventricular"]
        """
        Identify the fetal brain view.
        
        Parameters: 
            image (PIL.Image): Input Fetal Brain plane ultrasound image.
            
        Returns:
        Identified Brain view class.
        """

        # Preprocess image
        image = image.convert("L")
        image2 = PreprocessInput('bpc').preprocess_image(image)

        # Brain view prediction
        pred2 = bpc_model.predict(image2, verbose=0)
        confidence = round(np.max(pred2[0]), 2)
        pred2 = (pred2 > 0.2).astype('uint8')
        pred_view = np.argmax(pred2, axis=1)[0]
        brain_view = BPC_classes[pred_view]

        return brain_view, confidence
    
    def get_brain_biometry(self, binary_mask, img_size, pixel_size):
        """
        Calculate brain biometry from a binary mask.
        
        Parameters:
            binary_mask (numpy.ndarray): Binary mask of the brain region.
            size (tuple): Desired size for resizing the mask.
        
        Returns:
            tuple: Circumference, minor axis diameter, major axis diameter, mask, ellipse.
        """
        pixel_size = float(pixel_size)
        binary_mask = self.postprocess.process_binary(binary_mask)

        # Resize binary mask first
        resized_mask = cv2.resize(binary_mask, img_size, interpolation=cv2.INTER_NEAREST)

        # Apply ellipse fitting on the resized mask
        ellipse, pred_mask = self.ellipse_fitting.fit_ellipse_cv2(resized_mask)

        axes = ellipse[1]
        a = max(axes)
        b = min(axes)

        C = (np.pi / 2) * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        # C = 1.57 * (a + b)

        # Biometry measurement
       
        bpd = round((b - 0.04*b) *pixel_size * CALIBRATION_FACTOR, 2)
        ofd = round(a*pixel_size * CALIBRATION_FACTOR, 2)
        hc = round(C*pixel_size* CALIBRATION_FACTOR, 2)
     

        biometry = {"BPD": f'{bpd} cm',
                    "OFD": f'{ofd} cm',
                    "HC": f'{hc} cm'}

        return biometry, pred_mask, ellipse

    def fetal_brain_measurement(self, image, pixel_size):
        """
        Process the image and calculate fetal brain biometry.
        
        Parameters:
            image (PIL.Image): Input ultrasound image.
            pixel_size (float): Pixel size to compute real-world measurements.
        
        Returns:
            PIL.Image: Processed image with fitted ellipse and measurements.
        """
        image = image.convert("L")
        size = image.size

        # Preprocess the image
        image3 = PreprocessInput('hc').preprocess_image(image)
        pred3 = hc_model.predict(image3, verbose=0)
        pred3 = (pred3 > 0.2).astype('uint8').reshape((256, 256))

        biometry, _, ellipse_param = self.get_brain_biometry(pred3, size, pixel_size)

        return biometry, self.ellipse_fitting.render_fetal_brain_ellipse(image, ellipse_param)


    # Visualization function
    def visualize_and_return_image(self, image, title):
        """ 
        Plots the given image with the provided title
        """       
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        
        # Save the plot to a BytesIO buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to avoid display
        buf.seek(0)  # Rewind the buffer

        return Image.open(buf)  # Return the image from the buffer




