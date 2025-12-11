import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class EllipseFitting:
    def __init__(self):
        pass

    def fit_ellipse_cv2(self, binary_mask, show=False):
        """
        Fit an ellipse to the binary mask and return the ellipse parameters and mask.
        
        Parameters:
            binary_mask (numpy.ndarray): The binary mask to fit the ellipse.
            show (bool): Whether to show the mask and ellipse plot.
        
        Returns:
            tuple: Ellipse parameters, ellipse mask.
        """
        all_points = np.argwhere(binary_mask)
        all_points = all_points[:, [1, 0]]
        
        if len(all_points) >= 3:
            ellipse = cv2.fitEllipse(all_points)
        else:
            print("Not enough points to fit an ellipse!")
            return None, None

        ellipse_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        cv2.ellipse(ellipse_mask, ellipse, (1), 1)
        kernel = np.ones((2, 2), np.uint8)
        ellipse_mask = cv2.dilate(ellipse_mask, kernel, iterations=1)

        if show:
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))
            ax1.imshow(binary_mask, cmap='gray')
            ax1.set_title("Input Segmentation Mask")
            ax1.axis('off')

            ax3.imshow(ellipse_mask, cmap='gray')
            ax3.set_title("Fitted Ellipse")
            ax3.axis('off')

            plt.show()

        return ellipse, ellipse_mask

    def render_fetal_brain_ellipse(self, image, params):
        """
        Draw an ellipse on an image based on fitted parameters.
        
        Parameters:
            image (PIL.Image): The input image.
            params (tuple): The ellipse parameters.
            labels (list): Labels for the axes (HC, BPD, OFD).
            title (str): Title for the plot.
        
        Returns:
            PIL.Image: The image with the fitted ellipse and axes drawn.
        """
        center, axes, angle = params[0], params[1], params[2]

        if axes[0] < axes[1]:
            axes = (axes[1], axes[0])
            angle = (angle + 90) % 180

        a, b = axes[0], axes[1]
        theta = np.deg2rad(angle)

        t = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = center[0] + (a / 2) * np.cos(t) * np.cos(theta) - (b / 2) * np.sin(t) * np.sin(theta)
        y_ellipse = center[1] + (a / 2) * np.cos(t) * np.sin(theta) + (b / 2) * np.sin(t) * np.cos(theta)

        major_axis_start = (center[0] - a / 2 * np.cos(theta), center[1] - a / 2 * np.sin(theta))
        major_axis_end = (center[0] + a / 2 * np.cos(theta), center[1] + a / 2 * np.sin(theta))
        minor_axis_start = (center[0] - b / 2 * np.sin(theta), center[1] + b / 2 * np.cos(theta))
        minor_axis_end = (center[0] + b / 2 * np.sin(theta), center[1] - b / 2 * np.cos(theta))

        plt.imshow(image, cmap='gray')

        plt.plot(x_ellipse, y_ellipse, label='HC', color='blue', linestyle=":")
        plt.plot([minor_axis_start[0], minor_axis_end[0]], [minor_axis_start[1], minor_axis_end[1]], color='red', label='BPD', linestyle=":")
        plt.plot([major_axis_start[0], major_axis_end[0]], [major_axis_start[1], major_axis_end[1]], color='green', label='OFD', linestyle=":")

        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return Image.open(buf)
