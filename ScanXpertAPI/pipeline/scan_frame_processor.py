import os
from PIL import Image
from pipeline.brain_biometry import BrainBiometryPipeline, resource_path
from PIL import ImageDraw, ImageFont
import cv2
import numpy as np
from collections import Counter
from pipeline.femur_biometry import FemurLengthEstimation

# Fetal Brain Processing Pipeline
class USFrameProcess:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.classes = ['Abdomen', 'Femur', 'Cervix', 'Thorax', 'Brain', 'Other']
        self.brain_views = ["Trans-cerebellar", "Trans-thalamic", "Trans-ventricular"]
        self.images = self.load_images()
        self.classified_images = self.classify_images()
        self.brain_images = self.classify_brain_views()

    def load_images(self):
        """Loads images from the directory
        
        Returns:
            dict: Image name: Image
        """
        images = {
            img: Image.open(os.path.join(self.image_folder, img))
            for img in os.listdir(self.image_folder)
            if img.endswith(('.png', '.jpg', '.jpeg'))
        }
        return images

    def classify_images(self):
        """All the images undergo initial ultrasound plane classification

        Returns:
            dict: {Ultrasound Plane: Images (list)}
        """
        classified_images = {class_name: [] for class_name in self.classes}
        for _, img in self.images.items():
            plane, confidence = BrainBiometryPipeline().plane_classification(img)
            if confidence > 0.5:
                classified_images[plane].append((img, confidence))
        for class_name in classified_images:
            classified_images[class_name].sort(key=lambda x: x[1], reverse=True)
        return classified_images

    def classify_brain_views(self):
        """All the Brain plane images undergo brain view classification

        Returns:
            dict: {Brain View: Images (list)}
        """
        brain_images = {view: [] for view in self.brain_views}
        for im, _ in self.classified_images["Brain"]:
            view_plane, confidence1 = BrainBiometryPipeline().brain_view_classification(im)
            if confidence1 > 0.5:
                brain_images[view_plane].append((im, confidence1))
        for view_name in brain_images:
            brain_images[view_name].sort(key=lambda x: x[1], reverse=True)
        return brain_images

    def show_class_images(self, class_name):
        """Get all the images of a particular plane

        Args:
            class_name (string): Standard plane to be viewed

        Returns:
            list: (Image, Probability score)
        """
        images_with_confidence = self.classified_images[class_name]
        return [(img, f"Confidence: {confidence:.2f}") for img, confidence in images_with_confidence]

    def show_brain_images(self, view_name):
        """Get all the images of a particular brain view

        Args:
            view_name (string): Brain views to be seen

        Returns:
            list: (Image, Probability score)
        """
        images_with_confidence = self.brain_images[view_name]
        
        return [(img, f"Confidence: {confidence:.2f}") for img, confidence in images_with_confidence]
    
    def find_plane(self, image):
        """Identifies the plane view of the image

        Args:
            image (PILImage): The image to be measured

        Returns:
            string: Identified plane
        """
        plane, _ = BrainBiometryPipeline().plane_classification(image)
        if plane == "Brain":
            plane, _ = BrainBiometryPipeline().brain_view_classification(image)
        return plane

    def find_pixel_spacing(self, image_pil, cm_distance=1):
        """Computes the pixel spacing of the image

        Args:
            image_pil (PILImage): The image to be measured
            cm_distance (int, optional): Ruler spacing in cm. Defaults to 1.

        Returns:
            float: Pixel spacing rounded to 6 decimal places
        """
        # Convert PIL image to NumPy array (grayscale)
        image = np.array(image_pil.convert("L"))
        
        # Make a copy for processing
        processed_image = image.copy()
        
        # Mask the center region (keep only the ruler on the sides)
        processed_image[:, int(image.shape[1] * 0.15): int(image.shape[1] * 0.8)] = 0

        # Apply edge detection to detect tick marks
        edges = cv2.Canny(processed_image, 100, 255)

        # Find contours (assuming tick marks are vertical lines)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding box (x, y) positions for each contour
        tick_positions = [(cv2.boundingRect(cnt)[0], cv2.boundingRect(cnt)[1]) for cnt in contours]

        # Group by x-coordinates
        x_counts = {}
        for x, y in tick_positions:
            x_counts.setdefault(x, []).append(y)

        # Find y-coordinates where x-coordinates have multiple matches
        matched_y_ticks = []
        matched_x_tick = None
        for x, y_list in x_counts.items():
            if len(y_list) > len(matched_y_ticks) and x != 0:
                matched_y_ticks = sorted(y_list)
                matched_x_tick = x

        if not matched_y_ticks:
            print("No tick marks detected.")
            return None

        # Compute pixel distances between consecutive tick marks
        pixel_distances = np.diff(matched_y_ticks)

        # Find repeating distances
        distance_counts = Counter(pixel_distances)
        common_distances = [d for d, count in distance_counts.items() if count >= 2 and d < 50]

        if not common_distances:
            # Find the closest two distances
            min_diff = float('inf')
            pairs = []
            for i in range(len(pixel_distances) - 1):
                for j in range(i + 1, len(pixel_distances)):
                    diff = abs(pixel_distances[j] - pixel_distances[i])
                    if diff < min_diff:
                        pairs = [pixel_distances[j], pixel_distances[i]]
                        min_diff = diff

            common_distances.extend(pairs)
            if not common_distances or max(common_distances) < 8:
                print("No repeating distances found reliably.")
                return None

        # Use the largest detected repeating distance
        best_distance = max(common_distances) - 1

        # Calculate pixel spacing (pixels per cm)
        pixel_spacing = cm_distance / best_distance

        return round(pixel_spacing, 5)
            
    def measure(self, image, pixel_size, threshold=50):
        """Automatic measurement of Brain Biometry. Adds the obtained biometry as a textbox over the image.

        Args:
            image (Image): Image to be measured
            pixel_size (float): pixel spacing information (cm/px)
            threshold (int, optional): Connectivity pixels to filter the binary mask. Defaults to 50.

        Returns:
            tuple: Biometry label, Ellipse fitted image
        """
        biometry, im2 = BrainBiometryPipeline(threshold).fetal_brain_measurement(image, pixel_size)
        
        # Overlay biometry on the image
        draw = ImageDraw.Draw(im2)
        text = f"BPD: {biometry['BPD']}\nOFD: {biometry['OFD']}\nHC: {biometry['HC']}"
        font = ImageFont.truetype(resource_path("resources/times_new_roman_bold.ttf"), size=14)
        # Calculate text size
        text_bbox = draw.multiline_textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Define text position and background rectangle position
        text_position = (10, im2.height - text_height - 20)  # Adjust position for bottom left corner
        background_position = [
            text_position[0] - 5, 
            text_position[1] - 5, 
            text_position[0] + text_width + 10, 
            text_position[1] + text_height + 10
        ]

        # Draw black rectangle as background
        draw.rectangle(background_position, fill=(0, 0, 0))

        # Draw text on top of the rectangle
        draw.multiline_text(text_position, text, fill=(255, 255, 255), font=font)
        
        return biometry, im2
    
    def fl_measure(self, image, px, thresh, cm_distance=1):
        """Automatic measurement of Femur Length. Adds the obtained length as a textbox over the image.

        Args:
            image (Image): Image to be measured
            cm_distance (int, optional): Ruler spacing in cm. Defaults to 1.

        Returns:
            tuple: Femur length label, Image with length overlay
        """
        fl_length, output_image = FemurLengthEstimation(thresh).estimate_femur_length(image)

        femur_length = round(fl_length * float(px), 2) if fl_length is not None else None
        
        # Overlay femur length on the image
        draw = ImageDraw.Draw(output_image)
        text = f"FL: {femur_length} cm"
        font = ImageFont.truetype(resource_path("resources/times_new_roman_bold.ttf"), size=14)
        
        # Calculate text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Define text position and background rectangle position
        text_position = (10, image.height - text_height - 20)

        background_position = [
            text_position[0] - 5, 
            text_position[1] - 5, 
            text_position[0] + text_width + 10, 
            text_position[1] + text_height + 10
        ]
        # Draw black rectangle as background
        draw.rectangle(background_position, fill=(0, 0, 0))
        # Draw text on top of the rectangle
        draw.text(text_position, text, fill=(255, 255, 255), font=font)
        return femur_length, output_image