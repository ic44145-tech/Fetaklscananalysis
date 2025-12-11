import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
tf.config.set_visible_devices([], 'GPU')

class XceptionModel:
    
    def __init__(self, input_shape, num_classes, weight_path=None):
        """
        Initialize the XceptionModel class with input shape, number of classes, 
        and an optional path to pre-trained weights.
        
        Arguments:
        input_shape (tuple): The shape of the input data (height, width, channels).
        num_classes (int): Number of output classes.
        weight_path (str): Optional path to pre-trained weights.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_path = weight_path
        self.model = self.build_model()

    def build_model(self):
        """
        Build and compile the Xception model.
        
        Returns:
        model: Compiled Keras model.
        """
        # Initialize the Xception model without top layers
        base_model = Xception(include_top=False,
                              weights=None,
                              input_shape=self.input_shape,
                              pooling='avg')
        
        # Add a final dense layer for classification
        x = base_model.output
        outputs = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        
        # Compile the model
        loss = tf.keras.losses.categorical_crossentropy
        optimizer = RMSprop(learning_rate=0.0001)
        METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='acc'),
            tf.keras.metrics.Precision(name='pre'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
        model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

        # Load weights if the path is provided
        if self.weight_path:
            model.load_weights(self.weight_path)
            print(f"Loaded weights from {self.weight_path}")
        
        return model

    def get_model(self):
        """
        Get the compiled Xception model.
        
        Returns:
        model: Compiled Keras model.
        """
        return self.model
