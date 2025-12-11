from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Conv2DTranspose, concatenate
from keras.models import Model

class UNetModel:
    def __init__(self, input_shape, n_filters=16, dropout=0.1, batchnorm=True, weight_path=None):
        """
        Initialize the UNetModel class with the given hyperparameters.

        Arguments:
        input_shape (tuple): The shape of the input data (height, width, channels).
        n_filters (int): The base number of filters for the first layer. Defaults to 16.
        dropout (float): Dropout rate for the Dropout layers. Defaults to 0.1.
        batchnorm (bool): Whether to apply batch normalization after each convolutional block. Defaults to True.
        weight_path (str): Optional path to pre-trained weights.
        """
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.weight_path = weight_path
        self.model = self.build_model()

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it."""
        # First layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x

    def build_model(self):
        """Builds the U-Net model."""
        # Contracting Path
        input_img = Input(shape=self.input_shape, name='img')  # Use shape instead of tuple
        c1 = self.conv2d_block(input_img, self.n_filters * 1, batchnorm=self.batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(self.dropout)(p1)
        
        c2 = self.conv2d_block(p1, self.n_filters * 2, batchnorm=self.batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(self.dropout)(p2)
        
        c3 = self.conv2d_block(p2, self.n_filters * 4, batchnorm=self.batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(self.dropout)(p3)
        
        c4 = self.conv2d_block(p3, self.n_filters * 8, batchnorm=self.batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(self.dropout)(p4)
        
        c5 = self.conv2d_block(p4, self.n_filters * 16, batchnorm=self.batchnorm)
        
        # Expansive Path
        u6 = Conv2DTranspose(self.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(self.dropout)(u6)
        c6 = self.conv2d_block(u6, self.n_filters * 8, batchnorm=self.batchnorm)
        
        u7 = Conv2DTranspose(self.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(self.dropout)(u7)
        c7 = self.conv2d_block(u7, self.n_filters * 4, batchnorm=self.batchnorm)
        
        u8 = Conv2DTranspose(self.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(self.dropout)(u8)
        c8 = self.conv2d_block(u8, self.n_filters * 2, batchnorm=self.batchnorm)
        
        u9 = Conv2DTranspose(self.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(self.dropout)(u9)
        c9 = self.conv2d_block(u9, self.n_filters * 1, batchnorm=self.batchnorm)
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])

        # Compile the model
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Load weights if provided
        if self.weight_path:
            model.load_weights(self.weight_path)
            print(f"Weights loaded from {self.weight_path}")
        
        return model

    def get_model(self):
        """Returns the compiled U-Net model."""
        return self.model
