import os
import numpy as np
import tensorflow as tf
import argparse
import time
from typing import Tuple, Optional
from tqdm import tqdm

class PCEDNetPredictor:
    
    def __init__(self, model_path: str, target_scales: int = 16, target_features: int = 6):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the trained .h5 model file
            target_scales: Number of scales expected by the model
            target_features: Number of features per scale expected by the model
        """
        self.model_path = model_path
        self.target_scales = target_scales
        self.target_features = target_features
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained PCEDNet model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Successfully loaded model from {self.model_path}")
            print(f"Model expects {self.target_scales} scales with {self.target_features} features each")
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")
    
    def _read_ssm_file(self, ssm_path: str) -> np.ndarray:
        """
        Read SSM file and convert to numpy array
        
        Args:
            ssm_path: Path to the SSM file
            
        Returns:
            SSM data as numpy array of shape (n_points, n_scales, features_per_scale)
        """
        with open(ssm_path, 'r') as f:
            lines = f.readlines()
        
        # Read header information
        n_points, total_features = map(int, lines[0].split())
        features_per_scale = int(lines[1])
        n_scales = int(lines[3])
        
        print(f"SSM file info: {n_points} points, {total_features} total features, {n_scales} scales, {features_per_scale} features per scale")
        
        # Skip header lines and read data
        data_lines = lines[5:]
        
        # Parse data
        all_data = []
        points_read = 0
        
        for line in data_lines:
            if line.strip() and points_read < n_points:
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == total_features:
                        all_data.append(values)
                        points_read += 1
                except ValueError:
                    # Skip lines with missing or invalid features
                    continue
        
        if len(all_data) == 0:
            raise ValueError(f"No valid data points found in {ssm_path}")
        
        # Convert to numpy array and reshape
        data_array = np.array(all_data, dtype=np.float32)
        actual_points = data_array.shape[0]
        
        print(f"Successfully read {actual_points} points from SSM file")
        
        # Reshape to (n_points, n_scales, features_per_scale)
        try:
            reshaped_data = data_array.reshape((actual_points, n_scales, features_per_scale))
        except ValueError as e:
            print(f"Reshape error: {e}")
            # Truncate data to fit expected dimensions if necessary
            expected_total = actual_points * n_scales * features_per_scale
            if data_array.size > expected_total:
                data_array = data_array.flatten()[:expected_total]
                reshaped_data = data_array.reshape((actual_points, n_scales, features_per_scale))
            else:
                raise ValueError(f"Cannot reshape data: expected {expected_total} elements, got {data_array.size}")
        
        # Adapt to target format if necessary
        adapted_data = self._adapt_data_to_target_format(reshaped_data, n_scales, features_per_scale)
        
        return adapted_data
    
    def _adapt_data_to_target_format(self, data: np.ndarray, 
                                   original_scales: int, original_features: int) -> np.ndarray:
        """
        Adapt data from original format to target format expected by the model
        
        Args:
            data: Original data tensor [n_points, original_scales, original_features]
            original_scales: Number of scales in original data
            original_features: Number of features per scale in original data
            
        Returns:
            Adapted data tensor [n_points, target_scales, target_features]
        """
        # Adapt scales
        if original_scales != self.target_scales:
            if original_scales > self.target_scales:
                data = data[:, :self.target_scales, :]
                print(f"Truncated from {original_scales} to {self.target_scales} scales")
            else:
                raise ValueError(f"Original data has {original_scales} scales, but model expects {self.target_scales}")
        
        # Adapt features
        if original_features != self.target_features:
            if original_features > self.target_features:
                data = data[:, :, :self.target_features]
                print(f"Truncated from {original_features} to {self.target_features} features")
            else:
                raise ValueError(f"Original data has {original_features} features, but model expects {self.target_features}")
        
        return data
    
    def _prepare_model_input(self, ssm_data: np.ndarray) -> dict:
        """
        Prepare SSM data for model input format
        
        Args:
            ssm_data: SSM data of shape (n_points, n_scales, n_features)
            
        Returns:
            Dictionary with inputs for each scale
        """
        inputs = {}
        for i in range(self.target_scales):
            inputs[f'scale_input_{i}'] = ssm_data[:, i, :]
        
        return inputs
    
    def predict_batch(self, ssm_data: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Make predictions on SSM data in batches
        
        Args:
            ssm_data: SSM data of shape (n_points, n_scales, n_features)
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array of shape (n_points, n_classes)
        """
        n_points = ssm_data.shape[0]
        n_batches = (n_points + batch_size - 1) // batch_size
        
        predictions = []
        
        print(f"Making predictions on {n_points} points in {n_batches} batches...")
        
        for i in tqdm(range(n_batches), desc="Predicting"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            batch_data = ssm_data[start_idx:end_idx]
            batch_inputs = self._prepare_model_input(batch_data)
            
            # Make prediction
            batch_pred = self.model.predict(batch_inputs, verbose=0)
            predictions.append(batch_pred)
        
        # Concatenate all predictions
        all_predictions = np.concatenate(predictions, axis=0)
        
        return all_predictions
    
    def predict_file(self, ssm_path: str, output_path: str, 
                    batch_size: int = 1024, class_threshold: float = 0.5) -> None:
        """
        Predict edge labels for a single SSM file
        
        Args:
            ssm_path: Path to input SSM file
            output_path: Path to output label file
            batch_size: Batch size for prediction
            class_threshold: Threshold for binary classification (if model outputs probabilities)
        """
        print(f"Processing file: {ssm_path}")
        
        # Read SSM data
        ssm_data = self._read_ssm_file(ssm_path)
        
        # Remove NaN values
        nan_mask = np.isnan(ssm_data).any(axis=2).any(axis=1)
        valid_indices = ~nan_mask
        valid_data = ssm_data[valid_indices]
        
        if len(valid_data) == 0:
            raise ValueError("No valid data points found after removing NaN values")
        
        print(f"Processing {len(valid_data)} valid points (removed {np.sum(nan_mask)} NaN points)")
        
        # Make predictions
        predictions = self.predict_batch(valid_data, batch_size)
        
        # Prepare output labels - Initialize all as -1 (non-edge)
        n_original_points = len(ssm_data)
        output_labels = np.full(n_original_points, -1, dtype=np.int32)
        
        # Process predictions based on model output
        if predictions.shape[1] == 1:
            # Binary classification with sigmoid output
            # Convert to edge (0) vs non-edge (-1) format
            edge_predictions = (predictions[:, 0] > class_threshold)
            pred_labels = np.where(edge_predictions, 0, -1)  # 0 for edge, -1 for non-edge
        elif predictions.shape[1] == 2:
            # Two-class classification with softmax (assuming class 1 is edge)
            edge_predictions = np.argmax(predictions, axis=1)
            pred_labels = np.where(edge_predictions == 1, 0, -1)  # 0 for edge, -1 for non-edge
        elif predictions.shape[1] == 3:
            # Three-class classification (sharp-edge=1, smooth-edge=2, non-edge=0)
            # Combine sharp and smooth edges as edges (0), non-edge as (-1)
            class_predictions = np.argmax(predictions, axis=1)
            pred_labels = np.where((class_predictions == 1) | (class_predictions == 2), 0, -1)
        else:
            raise ValueError(f"Unexpected model output shape: {predictions.shape}")
        
        # Assign predictions to valid points
        output_labels[valid_indices] = pred_labels
        
        # Write output file
        self._write_label_file(output_path, output_labels, ssm_path)
        
        # Print statistics
        unique_labels, counts = np.unique(output_labels, return_counts=True)
        print(f"Prediction statistics:")
        for label, count in zip(unique_labels, counts):
            if label == -1:
                print(f"  Non-edge (-1): {count} points ({(count / len(output_labels)) * 100:.2f}%)")
            elif label == 0:
                print(f"  Edge (0): {count} points ({(count / len(output_labels)) * 100:.2f}%)")
            else:
                print(f"  Other ({int(label)}): {count} points ({(count / len(output_labels)) * 100:.2f}%)")
        
        print(f"Results saved to: {output_path}")
    
    def _write_label_file(self, output_path: str, labels: np.ndarray, source_file: str) -> None:
        """
        Write labels to output file in the specified format
        
        Args:
            output_path: Path to output label file
            labels: Array of predicted labels (-1 for non-edge, 0 for edge)
            source_file: Source SSM file path (for header comment)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Write labels (one per line, no header)
            for label in labels:
                f.write(f"{int(label)}\n")
    
    def predict_directory(self, input_dir: str, output_dir: str, 
                         batch_size: int = 1024, class_threshold: float = 0.5) -> None:
        """
        Predict edge labels for all SSM files in a directory
        
        Args:
            input_dir: Directory containing SSM files
            output_dir: Directory to save label files
            batch_size: Batch size for prediction
            class_threshold: Threshold for binary classification
        """
        # Find all SSM files
        ssm_files = [f for f in os.listdir(input_dir) if f.endswith('.ssm')]
        
        if not ssm_files:
            print(f"No SSM files found in {input_dir}")
            return
        
        print(f"Found {len(ssm_files)} SSM files to process")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        for ssm_file in tqdm(ssm_files, desc="Processing files"):
            ssm_path = os.path.join(input_dir, ssm_file)
            
            # Generate output filename (same as input but with different extension)
            base_name = os.path.splitext(ssm_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}.ply")
            
            try:
                self.predict_file(ssm_path, output_path, batch_size, class_threshold)
            except Exception as e:
                print(f"Error processing {ssm_file}: {e}")
                continue
        
        print(f"Completed processing. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PCEDNet Edge Detection Prediction")
    parser.add_argument("--model", required=True, help="Path to trained PCEDNet model (.h5 file)")
    parser.add_argument("--input", required=True, help="Path to input SSM file or directory")
    parser.add_argument("--output", required=True, help="Path to output label file or directory")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for prediction")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for binary models")
    parser.add_argument("--scales", type=int, default=16, help="Number of scales expected by model")
    parser.add_argument("--features", type=int, default=20, help="Number of features per scale expected by model")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        return
    
    # Initialize predictor
    print("Initializing PCEDNet predictor...")
    predictor = PCEDNetPredictor(
        model_path=args.model,
        target_scales=args.scales,
        target_features=args.features
    )
    
    # Run prediction
    start_time = time.time()
    
    if os.path.isfile(args.input):
        if args.input.endswith('.ssm'):
            predictor.predict_file(args.input, args.output, args.batch_size, args.threshold)
        else:
            print("Error: Input file must have .ssm extension")
            return
    elif os.path.isdir(args.input):
        predictor.predict_directory(args.input, args.output, args.batch_size, args.threshold)
    else:
        print("Error: Input must be a file or directory")
        return
    
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()