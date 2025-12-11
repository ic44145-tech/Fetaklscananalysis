import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

class PostProcessOutput:
    
    def __init__(self, threshold=50):
        """
        Initialize the class with default parameters.
        
        Arguments:
        threshold (int): Minimum area for connected regions to be retained.
        """
        self.threshold = threshold

    def process_binary(self, bin_image, show=False):
        """
        Process a binary image to retain only connected regions above a certain threshold.
        
        Arguments:
        bin_image: The input binary image.
        show (bool): Whether to display the original and processed images.

        Returns:
        new_binary_image: The processed binary image with connected regions.
        """
        # Label connected regions
        labeled_image = label(bin_image, connectivity=2)

        # Get the coordinates of connected pixels
        regions = regionprops(labeled_image)

        # Create an empty binary image of the same size
        new_binary_image = np.zeros_like(bin_image)

        # Set the threshold for connected region size
        t = self.threshold

        # Fill the new binary image with connected pixels that meet the threshold
        for region in regions:
            if region.area > t:
                for coord in region.coords:
                    new_binary_image[coord[0], coord[1]] = 1

        # Optionally, display the original and new binary images
        if show:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            # Original binary image
            axes[0].imshow(bin_image, cmap='gray')
            axes[0].set_title("Original Binary Image")

            # Binary image with only connected pixels
            axes[1].imshow(new_binary_image, cmap='gray')
            axes[1].set_title("Connected Pixels Only")
            plt.show()

        return new_binary_image