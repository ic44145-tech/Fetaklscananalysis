import numpy as np
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input

class PreprocessInput:
    
    def __init__(self, model_type):
        """
        Initialize the class with the model type.
        
        Arguments:
        model_type (str): The type of model (e.g., 'uspc', 'bpc', 'hc')
        """
        self.model_type = model_type

    def preprocess_image(self, image):
        """
        Preprocess the input image based on the model type.

        Arguments:
        image: Input image to be preprocessed

        Returns:
        Preprocessed image
        """
        # Convert image to array
        image = img_to_array(image)

        # Resize the image to 256x256
        image = resize(image, (256, 256), mode='constant', preserve_range=True)
        
        # Apply model-specific preprocessing
        if self.model_type == 'uspc':
            image = preprocess_input(image)
            # image = image / 255.0
        elif self.model_type == 'bpc':
            image = np.repeat(image, 3, axis=-1)  # Repeat channels to get a 3-channel image
            image = image / 255.0  # Normalize to [0, 1] range
        elif self.model_type == 'hc':
            image = image / 255.0  # Normalize to [0, 1] range

        # Expand dimensions to add batch size
        image = np.expand_dims(image, axis=0)
        
        return image