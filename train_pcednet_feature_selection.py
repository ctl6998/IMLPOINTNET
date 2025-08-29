import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING messages
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Turn off GPU

import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, jaccard_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

from utils import DataGenerator, analyze_dataset, analyze_full_dataset
from models import PCEDNet


def calculate_metrics(y_true, y_pred):
    """
    Calculate all required metrics: Precision, Recall, MCC, IoU, F1-score
    """
    # Convert probabilities to binary predictions if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_binary = np.argmax(y_pred, axis=1)
    else:
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    y_true_binary = y_true.flatten().astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
    iou = jaccard_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Confusion matrix for additional insights
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    return {
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'iou': iou,
        'f1_score': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }


def evaluate_model(model, data_generator, dataset_name="Dataset"):
    """
    Evaluate model on a dataset and calculate all metrics
    """
    print(f"\n== Evaluating on {dataset_name} ==")
    
    all_predictions = []
    all_labels = []
    
    # Get predictions for all batches
    for i in range(len(data_generator)):
        batch_inputs, batch_labels = data_generator[i]
        batch_predictions = model.predict(batch_inputs, verbose=0)
        
        all_predictions.append(batch_predictions)
        all_labels.append(batch_labels)
    
    # Concatenate all predictions and labels
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(labels, predictions)
    
    # Print results
    print(f"{dataset_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  MCC:       {metrics['mcc']:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  F1-score:  {metrics['f1_score']:.4f}")
    print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    return metrics


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
    train_path = "/home/cle/Work/ABC-Challenge/Dataset/Train"
    val_path = "/home/cle/Work/ABC-Challenge/Dataset/Validation"
    
    # Model configuration
    selected_descriptors = [19, 17, 3, 4, 1, 16, 18, 8]  # Select which descriptors to use (1-based indexing)
    TARGET_SCALES = 16  # Max 16, must be power of 2 to match PCEDNet architecture
    BATCHES = 2048
    EPOCHS = 10
    LEARNING_RATE = 0.001
    FRACTION = 1.0  # Use full dataset
    
    # Create save directory
    save_dir = "/home/cle/Work/ABC-Challenge/Results"
    os.makedirs(save_dir, exist_ok=True)
    
    descriptor_str = "_".join(map(str, selected_descriptors))
    model_name = f"pcednet_desc{descriptor_str}_{TARGET_SCALES}s_{len(selected_descriptors)}f_{BATCHES}b_{EPOCHS}e_lr{str(LEARNING_RATE).replace('.', '')}"

    print("== Train data generation with balanced batches ==")
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
    
    print("== Validation data generation with balanced batches ==")
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

    # Test batch balance
    print("\n== Testing batch balance ==")
    sample_inputs, sample_labels = train_generator[0]
    unique, counts = np.unique(sample_labels, return_counts=True)
    print(f"Training batch class distribution:")
    for cls, count in zip(unique, counts):
        percentage = (count / len(sample_labels)) * 100
        print(f"  Class {cls}: {count} points ({percentage:.2f}%)")
    
    print(f"Input shape for scale_input_0: {sample_inputs['scale_input_0'].shape}")
    print(f"Number of scale inputs: {len([k for k in sample_inputs.keys() if k.startswith('scale_input_')])}")
    
    print("\n== Building model ==")
    pcednet = PCEDNet(
        num_scales=TARGET_SCALES, 
        features_per_scale=len(selected_descriptors)  # Use number of selected descriptors
    )
    model = pcednet.build_model()
    pcednet.compile_model(learning_rate=LEARNING_RATE)

    print("== Model Summary ==")
    pcednet.summary()
    
    print("== Starting training ==")
    training_start = time.perf_counter()
    
    # Train with validation
    history = pcednet.train(
        train_generator=train_generator,
        validation_generator=val_generator,
        epochs=EPOCHS
    )
    
    training_end = time.perf_counter()
    training_time = training_end - training_start
    print(f"\n== Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes) ==")
    
    # Final evaluation
    print("\n== Final Evaluation ==")
    
    # Evaluate on training set
    # train_metrics = evaluate_model(model, train_generator, "Training Set")
    
    # Evaluate on validation set
    val_metrics = evaluate_model(model, val_generator, "Validation Set")
    
    # Save model
    model_path = os.path.join(save_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"\n== Model saved to {model_path} ==")
    
    # Save training history and metrics
    results = {
        'model_config': {
            'selected_descriptors': selected_descriptors,
            'target_scales': TARGET_SCALES,
            'batch_size': BATCHES,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'training_time': training_time
        },
        'training_history': {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'val_loss': history.history.get('val_loss', []),
            'val_accuracy': history.history.get('val_accuracy', [])
        },
        'final_metrics': {
            # 'train': train_metrics,
            'validation': val_metrics
        }
    }
    
    # Save results to file
    results_path = os.path.join(save_dir, f"{model_name}_results.txt")
    with open(results_path, 'w') as f:
        f.write("=== TRAINING RESULTS ===\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Selected descriptors: {selected_descriptors}\n")
        f.write(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
        
        f.write("=== FINAL METRICS ===\n")
        # f.write("Training Set:\n")
        # for key, value in train_metrics.items():
        #     f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nValidation Set:\n")
        for key, value in val_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    
    print(f"== Results saved to {results_path} ==")
    print("\n=== TRAINING SUMMARY ===")
    print(f"Model: {model_name}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Validation F1-score: {val_metrics['f1_score']:.4f}")
    print(f"Validation IoU: {val_metrics['iou']:.4f}")
    print(f"Validation MCC: {val_metrics['mcc']:.4f}")


if __name__ == "__main__":
    # Dataset analysis
    print("\n=== Dataset Analysis ===")
    print("\n== Train Dataset Analysis ==")
    analyze_full_dataset("/home/cle/Work/ABC-Challenge/Dataset/Train")
    print("\n== Validation Dataset Analysis ==")
    analyze_full_dataset("/home/cle/Work/ABC-Challenge/Dataset/Validation")

    print("\n=== Starting Training with Descriptor Selection ===")
    main()