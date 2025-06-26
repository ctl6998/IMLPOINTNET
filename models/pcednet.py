import tensorflow as tf
import math


class PCEDNet:
    """
    PCEDNet implementation that can handle different numbers of scales and features
    """
    
    def __init__(self, num_scales: int = 16, features_per_scale: int = 6):
        self.model = None
        self.num_scales = num_scales
        self.features_per_scale = features_per_scale
        
        # num_scales value must be a power of 2 for proper tree reduction
        if num_scales & (num_scales - 1) != 0:
            raise ValueError(f"num_scales must be a power of 2, got {num_scales}")
    
    # Number of reduction layer is defined with flexibility
    def _calculate_tree_stages(self):
        """Calculate the number of tree reduction stages needed"""
        return int(math.log2(self.num_scales))
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the PCEDNet model
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = []
        for i in range(self.num_scales):
            input_layer = tf.keras.layers.Input(
                shape=(self.features_per_scale,), 
                name=f'scale_input_{i}'
            )
            inputs.append(input_layer)
        
        # Tree reduction stages 
        current_vectors = inputs
        num_stages = self._calculate_tree_stages()
        
        for stage in range(num_stages - 1):  # Stop before we get to 1 vector! The final stage is concatenation
            stage_outputs = []
            vectors_in_stage = len(current_vectors)
            
            for i in range(0, vectors_in_stage, 2):
                # Concatenate pairs of vectors
                concat = tf.keras.layers.Concatenate(
                    axis=-1, 
                    name=f'concat_stage{stage}_{i//2}'
                )([current_vectors[i], current_vectors[i+1]])
                
                # Fusion back to original feature size
                dense = tf.keras.layers.Dense(
                    self.features_per_scale,
                    activation='selu',
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.LecunUniform(),
                    bias_initializer=tf.keras.initializers.LecunUniform(),
                    name=f'fusion_stage{stage}_{i//2}'
                )(concat)
                
                stage_outputs.append(dense)
            
            current_vectors = stage_outputs
        
        # Final fusion: should have 2 vectors, each one is (features number x 1)
        current_vectors = current_vectors[:2]
        
        final_concat = tf.keras.layers.Concatenate(
            axis=-1, 
            name='final_concat'
        )(current_vectors)
        
        final_fusion = tf.keras.layers.Dense(
            self.features_per_scale * 2,  # Keep the concatenated size
            activation='selu',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            bias_initializer=tf.keras.initializers.LecunUniform(),
            name='final_fusion'
        )(final_concat)
        
        # First dense layer
        dense1 = tf.keras.layers.Dense(
            16,
            activation='selu',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            bias_initializer=tf.keras.initializers.LecunUniform(),
            name='dense1'
        )(final_fusion)
        
        # Batch normalization
        bn1 = tf.keras.layers.BatchNormalization(name='bn1')(dense1)
        
        # Second dense layer
        dense2 = tf.keras.layers.Dense(
            16,
            activation='selu',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            bias_initializer=tf.keras.initializers.LecunUniform(),
            name='dense2'
        )(bn1)
        
        # Second batch normalization
        bn2 = tf.keras.layers.BatchNormalization(name='bn2')(dense2)
        
        # Output layer with sigmoid activation
        output = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='output'
        )(bn2)
        
        # Create model
        self.model = tf.keras.Model(
            inputs=inputs, 
            outputs=output, 
            name=f'PCEDNet_{self.num_scales}s_{self.features_per_scale}f'
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