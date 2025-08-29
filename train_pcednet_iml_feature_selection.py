import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING messages
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Turn off GPU

import tensorflow as tf
import time
import numpy as np

from utils import DataGenerator, analyze_dataset, analyze_full_dataset
from models import PCEDNet

class ModifiedDataGenerator(DataGenerator):
    """
    Modified DataGenerator to support descriptor selection
    """
    def __init__(self, selected_descriptors=[1,2,3], **kwargs):
        """
        Args:
            selected_descriptors: List of descriptor indices (1-based) to extract
            **kwargs: Other arguments passed to parent class
        """
        self.selected_descriptors = selected_descriptors
        self.num_selected_descriptors = len(selected_descriptors)
        
        # Modify target_features to match selected descriptors
        kwargs['target_features'] = self.num_selected_descriptors
        
        super().__init__(**kwargs)
        
        print(f"Selected descriptors: {selected_descriptors}")
        print(f"Number of selected descriptors: {self.num_selected_descriptors}")
    
    def _adapt_data_to_target_format(self, data: np.ndarray, 
                                   original_scales: int, original_features: int) -> np.ndarray:
        """
        Adapt data from original format to target format with descriptor selection
        Args:
            data: Original data tensor [n_points, original_scales, original_features=20]
        Returns:
            Adapted data tensor [n_points, target_scales, num_selected_descriptors]
        """
        n_points = data.shape[0]
        
        # Handle scales (same as before)
        if original_scales != self.target_scales:
            if original_scales > self.target_scales:
                data = data[:, :self.target_scales, :]
            else:
                raise ValueError(f"Maximum scale in dataset is {original_scales}")
        
        # Handle descriptor selection
        if original_features == 20:  # Expected number of descriptors
            # Extract only selected descriptors (convert from 1-based to 0-based indexing)
            selected_indices = [d - 1 for d in self.selected_descriptors if 1 <= d <= 20]
            data = data[:, :, selected_indices]
            print(f"Extracted descriptors {self.selected_descriptors} from {original_features} original descriptors")
        else:
            raise ValueError(f"Expected 20 descriptors but found {original_features}")
        
        return data
    
    def _read_ssm_file(self, filepath: str) -> np.ndarray:
        """
        Read SSM file with descriptor selection capability
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Read header information
        n_points, total_features = map(int, lines[0].split())
        features_per_scale = int(lines[1])  # Should be 20
        n_scales = int(lines[3])  # Should be 16
        
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
                    continue
        
        if len(all_data) == 0:
            raise ValueError(f"No valid data points found in {filepath}")
        
        # Convert to numpy array and reshape
        data_array = np.array(all_data, dtype=np.float32)
        actual_points = data_array.shape[0]
        
        # Reshape to (n_points, n_scales, features_per_scale)
        # The data is arranged as: d1.1, d2.1, ..., d20.1, d1.2, d2.2, ..., d20.2, ..., d1.16, d2.16, ..., d20.16
        try:
            reshaped_data = data_array.reshape((actual_points, n_scales, features_per_scale))
        except ValueError as e:
            print(f"Reshape error: {e}")
            expected_total = actual_points * n_scales * features_per_scale
            if data_array.size > expected_total:
                data_array = data_array.flatten()[:expected_total]
                reshaped_data = data_array.reshape((actual_points, n_scales, features_per_scale))
            else:
                raise ValueError(f"Cannot reshape data: expected {expected_total} elements, got {data_array.size}")
        
        # Apply descriptor selection and scale adaptation
        adapted_data = self._adapt_data_to_target_format(
            reshaped_data, n_scales, features_per_scale
        )
        
        return adapted_data


def main(): 
    # Configuration
    # train_path = "/home/cle/Work/ABC-Challenge/Train"
    # val_path = "/home/cle/Work/ABC-Challenge/Validation"

    # IML Testing
    selected_descriptors = [18, 10, 2, 12] 
    base_path = "/home/cle/data/dtu_results_fs"
    save_path = f"{base_path}/IML_scan24/{'_'.join(map(str, selected_descriptors))}"
    train_path = "/home/cle/data/dtu_results_pc/IML_scan24"
    val_path = "/home/cle/data/dtu_results_pc/IML_scan24"
    TARGET_SCALES = 16  # Max 16, must be power of 2 to match PCEDNet architecture
    # TARGET_FEATURES = 16  # Max 20
    SCALE_SIZE = 15 # 8% // 15% (default) diagonal
    BATCHES = 512
    EPOCHS = 50
    FRACTION = 1 #0.1 (train only 10%) or 1 (train all)
    USE_VALIDATION = False

    print("== Train data generating with balanced batches ==")
    train_generator = ModifiedDataGenerator(
        base_path=train_path,
        selected_descriptors=selected_descriptors,
        batch_size=BATCHES, 
        shuffle=True,
        target_scales=TARGET_SCALES,
        balance_classes=True, 
        num_classes=2,
        dataset_fraction=FRACTION,
        output_format='dict'
    )
    print(f"Training batches: {len(train_generator)}")
    
    val_generator = None
    if USE_VALIDATION:
        print("== Validation data generating with balanced batches ==")
        val_generator = ModifiedDataGenerator(
            base_path=val_path,
            selected_descriptors=selected_descriptors,
            batch_size=BATCHES,
            shuffle=False,
            target_scales=TARGET_SCALES,
            balance_classes=True,
            num_classes=2,
            dataset_fraction=FRACTION,
            output_format='dict'
        )
        print(f"Validation batches: {len(val_generator)}")
    else:
        print("== Validation disabled for faster training ==")

    # Test a batch to see class distribution
    print("\n== Testing batch balance ==")
    for i in range(min(1, len(train_generator))):  
        sample_inputs, sample_labels = train_generator[i]
        unique, counts = np.unique(sample_labels, return_counts=True)
        print(f"Batch {i} class distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / len(sample_labels)) * 100
            print(f"  Class {cls}: {count} points ({percentage:.2f}%)")
        print()
    
    print("\n== Building model ==")
    pcednet = PCEDNet(
        num_scales=TARGET_SCALES, 
        features_per_scale=TARGET_FEATURES
    )
    model = pcednet.build_model()
    pcednet.compile_model(learning_rate=0.01) #0.001 OR 0.01

    print("== Model Summary ==")
    pcednet.summary()
    
    print("== Starting training ==")
    training_start = time.perf_counter()
    history = pcednet.train(
        train_generator=train_generator,
        validation_generator=val_generator,
        epochs=EPOCHS
    )
    training_end = time.perf_counter()
    training_time = training_end - training_start
    print(f"\n== Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes) ==")
    

    print("== Save model ==")
    model.save(f"{save_path}/IML_pcednet_{TARGET_SCALES}s_{TARGET_FEATURES}f_{BATCHES}b_{EPOCHS}e_lr01_size{SCALE_SIZE}.h5")
    
    print("Training completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    if USE_VALIDATION and 'val_accuracy' in history.history:
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    # Uncomment to analyze datasets first
    print("\n=== Dataset Analysis ===")
    print("\n== Train Dataset Analysis ==")
    analyze_full_dataset("/home/cle/data/dtu_results_pc/IML_scan24")
    # print("\n== Validation Dataset Analysis ==")
    # analyze_dataset("/home/cle/Work/ABC-Challenge/Validation")

    print("\n=== Starting Training with Balanced Batches ===")
    main()