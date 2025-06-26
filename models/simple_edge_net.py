import tensorflow as tf


class SimpleEdgeNet:
    """
    Simple Neural Network for edge detection with flattened input
    """
    
    def __init__(self, num_scales: int = 16, features_per_scale: int = 6):
        self.model = None
        self.num_scales = num_scales
        self.features_per_scale = features_per_scale
        self.input_size = num_scales * features_per_scale  # Flattened input size
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the simple neural network model
        Returns:
            Compiled Keras model
        """
        # Input layer - flattened input
        input_layer = tf.keras.layers.Input(
            shape=(self.input_size,), 
            name='flattened_input'
        )
        
        # Hidden layer with 60 neurons and SELU activation
        hidden = tf.keras.layers.Dense(
            60,
            activation='selu',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            bias_initializer=tf.keras.initializers.LecunUniform(),
            name='hidden_layer'
        )(input_layer)
        
        # Output layer with sigmoid activation
        output = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='output'
        )(hidden)
        
        # Create model
        self.model = tf.keras.Model(
            inputs=input_layer, 
            outputs=output, 
            name=f'SimpleEdgeNet_{self.num_scales}s_{self.features_per_scale}f'
        )
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile the model with optimizer and metrics
        
        Args:
            learning_rate: Initial learning rate for Adam optimizer
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Custom metrics
        def precision(y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
            return true_positives / (predicted_positives + tf.keras.backend.epsilon())
        
        def recall(y_true, y_pred):
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
            return true_positives / (possible_positives + tf.keras.backend.epsilon())
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', precision, recall]
        )
    
    def get_callbacks(self, epochs: int):
        """
        Get training callbacks
        Args:
            epochs: Total number of training epochs
        Returns:
            List of callbacks
        """
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            factor=tf.math.exp(-4.1),
            patience=int(epochs * 2 / 10),
            min_lr=0.0000001,
            verbose=1
        )
        
        return [reduce_lr]
    
    def train(self, train_generator, validation_generator, epochs: int = 50):
        """
        Train the model
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        callbacks = self.get_callbacks(epochs)
        
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def summary(self):
        """
        Print model summary
        """
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nParameter Summary:")
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")
        print(f"Configuration: {self.num_scales} scales, {self.features_per_scale} features per scale")
        print(f"Flattened input size: {self.input_size}")