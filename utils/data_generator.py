import os
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Generator
from tqdm import tqdm

SCALE_DEFAULT = 16
FEATURE_DEFAULT = 20
BATCH_DEFAULT = 1024


class DataGenerator(tf.keras.utils.Sequence):
    """
    Loading SSM and label files in batches with balanced sampling
    Can handle variable number of scales and features
    Supports both dictionary output (for PCEDNet) and flattened output (for SimpleEdgeNet)
    """
    def __init__(
        self, 
        base_path: str, 
        batch_size: int = BATCH_DEFAULT, 
        shuffle: bool = True, 
        split: int = 1, 
        target_scales: int = SCALE_DEFAULT, 
        target_features: int = FEATURE_DEFAULT,
        balance_classes: bool = True,
        num_classes: int = 3,  # 0: non-edge, 1: sharp-edge, 2: smooth-edge
        output_format: str = 'dict',  # 'dict' for PCEDNet, 'flat' for SimpleEdgeNet
        dataset_fraction: float = 1.0  # New parameter: fraction of dataset to use (0.1 = 10%)
        ):
        self.base_path = base_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_scales = target_scales
        self.target_features = target_features
        self.balance_classes = balance_classes
        self.num_classes = num_classes
        self.output_format = output_format
        self.dataset_fraction = max(0.01, min(1.0, dataset_fraction))  # Clamp between 1% and 100%
        
        self.ssm_path = os.path.join(base_path, "SSM_Challenge-ABC")
        self.lb_path = os.path.join(base_path, "lb")
        self.ply_path = os.path.join(base_path, "ply")
        
        self.file_list = []
        self._load_file_list()
        
        # For balanced sampling
        self.all_data = None
        self.all_labels = None
        self.class_indices = {}  # Dictionary to store indices for each class
        self.indices_pool = []   # Pool of balanced indices for sampling
        
        self._load_all_data()
        if self.balance_classes:
            self._prepare_balanced_indices()
        
        self.split = split
        self.current_index = 0
    
    def _load_file_list(self):
        """Load list of matching SSM and LB files"""
        if not os.path.exists(self.ssm_path):
            raise FileNotFoundError(f"SSM directory not found: {self.ssm_path}")
        if not os.path.exists(self.lb_path):
            raise FileNotFoundError(f"Label directory not found: {self.lb_path}")
            
        ssm_files = [f for f in os.listdir(self.ssm_path) if f.endswith('.ssm')]
        
        # For reducing the dataset
        if self.dataset_fraction < 1.0:
            original_count = len(ssm_files)
            target_count = max(1, int(len(ssm_files) * self.dataset_fraction))
            ssm_files = ssm_files[:target_count]
            print(f"Using {len(ssm_files)} files out of {original_count} ({self.dataset_fraction*100:.1f}% of dataset)")
        
        for ssm_file in ssm_files:
            base_name = ssm_file.replace('.ssm', '')
            lb_file = base_name + '.lb'
            
            ssm_full_path = os.path.join(self.ssm_path, ssm_file)
            lb_full_path = os.path.join(self.lb_path, lb_file)
            
            # Check the corresponding lb and ssm
            if os.path.exists(lb_full_path):
                self.file_list.append({
                    'base_name': base_name,
                    'ssm_file': ssm_full_path,
                    'lb_file': lb_full_path
                })
        
        print(f"Found {len(self.file_list)} matching SSM-LB file pairs")
    
    def _adapt_data_to_target_format(self, data: np.ndarray, 
                                   original_scales: int, original_features: int) -> np.ndarray:
        """
        Adapt data from original format to target format
        Args:
            ABC dataset: Original data tensor [n_points, original_scales=20, original_features=16]         
        Returns:
            Adapted dataset: new data tensor [n_points, target_scales, target_features] -> in numpy arrary
        """
        n_points = data.shape[0]
        
        if original_scales != self.target_scales:
            if original_scales > self.target_scales:
                data = data[:, :self.target_scales, :]
                # print(f"Truncated from {original_scales} to {self.target_scales} scales")
            else:
                raise ValueError(f"Maximum scale in dataset is {SCALE_DEFAULT}")
        
        if original_features != self.target_features:
            if original_features > self.target_features:
                data = data[:, :, :self.target_features]
                # print(f"Truncated from {original_features} to {self.target_features} features")
            else:
                raise ValueError(f"Maximum features in dataset is {FEATURE_DEFAULT}")
        
        return data
    
    def _read_ssm_file(self, filepath: str) -> np.ndarray:
        """
        Read SSM file and convert to numpy array
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Read the zero line: number of points and total features
        n_points, total_features = map(int, lines[0].split()) #total features = 20 features x 16 scales = 320
        # Read the first line: number of features
        features_per_scale = int(lines[1]) # 20
        # Read the third line: number of scales
        n_scales = int(lines[3]) # 16
        # print(f"File format show: {n_points} points, {total_features} total features, {n_scales} scales, {features_per_scale} features")
        
        # Skip header lines and read data
        data_lines = lines[5:] 

        # Parse data into all_data array
        all_data = []
        points_read = 0
        for line in data_lines:
            if line.strip() and points_read < n_points:
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == total_features:  
                        all_data.append(values)
                        points_read += 1
                except ValueError: # Skip lines (point) with missin features (not enough)
                    continue
        
        if len(all_data) == 0:
            raise ValueError(f"No valid data points found in {filepath}")
        
        # Convert all_data to numpy array
        data_array = np.array(all_data, dtype=np.float32)
        actual_points = data_array.shape[0]
        # print(f"Successfully read {actual_points} points")
        
        # Reshape to (n_points, n_scales, features_per_scale)
        try:
            reshaped_data = data_array.reshape((actual_points, n_scales, features_per_scale))
        except ValueError as e:
            print(f"Reshape error: {e}")
            # Truncate data to fit expected dimensions
            expected_total = actual_points * n_scales * features_per_scale
            if data_array.size > expected_total:
                data_array = data_array.flatten()[:expected_total]
                reshaped_data = data_array.reshape((actual_points, n_scales, features_per_scale))
            else:
                raise ValueError(f"Cannot reshape data: expected {expected_total} elements, got {data_array.size}")
        
        # Adapt to target format
        adapted_data = self._adapt_data_to_target_format(
            reshaped_data, n_scales, features_per_scale
        )
        
        return adapted_data
    
    def _read_lb_file(self, filepath: str) -> np.ndarray:
        """
        Read label file and convert to numpy array
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        labels = []
        for line in lines[1:]:
            if line.strip():
                try:
                    labels.append(float(line.strip()))
                except ValueError: # Skip invalid lines
                    continue
        
        return np.array(labels, dtype=np.float32)
    
    def _load_all_data(self):
        """Load all data into memory for balanced sampling"""
        print("Loading all data into memory for balanced sampling...")
        all_data_list = []
        all_labels_list = []
        total_points = 0
        filtered_points = 0
        non_annotated_excluded = 0

        print(f"Found {len(self.file_list)} files to process")
        
        for file_info in tqdm(self.file_list, desc="Loading files", unit="file"):
            try:
                # Load SSM and labels data
                data = self._read_ssm_file(file_info['ssm_file'])
                labels = self._read_lb_file(file_info['lb_file'])
                
                # Truncating to ensure data and labels have same length
                min_length = min(len(data), len(labels))
                data = data[:min_length]
                labels = labels[:min_length]

                total_points += len(labels)
                
                # Remove NaN values
                nan_mask = np.isnan(data).any(axis=2).any(axis=1)
                valid_data = data[~nan_mask]
                valid_labels = labels[~nan_mask]

                # Filter out non-annotated points (label = -1)
                # Only keep annotated points: 0 (non-edge) and 1 (edge) #BINARY ONLY
                # May need condition
                annotated_mask = (valid_labels != -1)
                annotated_data = valid_data[annotated_mask]
                annotated_labels = valid_labels[annotated_mask]

                #print(f"  After filtering -1: data shape: {annotated_data.shape}, labels length: {len(annotated_labels)}") 
                #print(f"  Annotated label distribution: {np.unique(annotated_labels, return_counts=True)}") 
            
                filtered_points += len(annotated_labels)
                non_annotated_excluded += len(valid_labels) - len(annotated_labels)
                
                # all_data_list.append(valid_data)
                # all_labels_list.append(valid_labels)
                if len(annotated_data) > 0:
                    all_data_list.append(annotated_data)
                    all_labels_list.append(annotated_labels)
                    print(f"Added {len(annotated_data)} points to the memory")
                else:
                    print(f"No annotated data found in this file!")
                
                print(f"Loaded {file_info['base_name']}: {len(annotated_data)} annotated points; (excluded {len(valid_labels) - len(annotated_data)} non-annotated)")
                
                tqdm.write(f"Loaded {file_info['base_name']}: {len(valid_data)} points")
            
            except Exception as e:
                print(f"Error loading file {file_info['base_name']}: {e}")
                continue

        # Analyze how many points, how many points per class
        if all_data_list:
            self.all_data = np.concatenate(all_data_list, axis=0)
            self.all_labels = np.concatenate(all_labels_list, axis=0)
        
            print(f"\n=== Data Loading Summary ===")
            print(f"Total points in dataset: {total_points}")
            print(f"Non-annotated points (excluded): {non_annotated_excluded}")
            print(f"Annotated points (kept for training): {filtered_points}")
            print(f"Filtering ratio: {filtered_points/total_points*100:.2f}%")
        
            # Analyze class distribution
            unique, counts = np.unique(self.all_labels, return_counts=True)
            print(f"\n=== Class Distribution ===")
            for cls, count in zip(unique, counts):
                percentage = (count / len(self.all_labels)) * 100
                class_name = "non-edge" if cls == 0 else "edge"
                print(f"Class {int(cls)} ({class_name}): {count} points ({percentage:.2f}%)")
        else:
            raise ValueError("No valid data loaded")
    
    def _prepare_balanced_indices(self):
        """Prepare balanced points data for sampling batches"""
        print("== Preparing balanced points... ==")
        
        for class_id in range(self.num_classes): # Iterate through 0,1 
            class_mask = (self.all_labels == class_id) # Boolean array [T,F,T...], T for where label = class_id
            self.class_indices[class_id] = np.where(class_mask)[0] # Assign to class_indices: key [class_id] where class mark value = True
            """
            self.class_indices = {
                0: array([0, 1, 2, 5, 8, 10, ...]),     # label 0 at these positions
                1: array([3, 4, 6, 7, 9, 11, ...])      # label 1 at these positions
            }
            """
            print(f"Class {class_id}: {len(self.class_indices[class_id])} points")
        
        # Decide which class have larger point (majority), other classes to be replicated
        max_class_size = max(len(indices) for indices in self.class_indices.values())
        print(f"Maximum class size: {max_class_size}")
        
        # Create balanced pool by replicating minority classes
        balanced_indices = []
        for class_id in range(self.num_classes):
            class_indices = self.class_indices[class_id]
            if len(class_indices) == 0:
                continue
                
            replications_needed = max_class_size // len(class_indices)
            remainder = max_class_size % len(class_indices)
            
            replicated_indices = np.tile(class_indices, replications_needed)
            if remainder > 0: # Add random class_indices for remainder
                additional_indices = np.random.choice(class_indices, remainder, replace=False)
                replicated_indices = np.concatenate([replicated_indices, additional_indices])
            
            balanced_indices.extend(replicated_indices)
            print(f"Class {class_id}: replicated to {len(replicated_indices)} points")
        
        self.indices_pool = np.array(balanced_indices)
        print(f"Total balanced pool size: {len(self.indices_pool)}")
        print("== Finish Preparing balanced points... ==")
    
    def __len__(self):
        """Calculate number of batches"""
        if self.balance_classes and self.indices_pool is not None:
            total_points = len(self.indices_pool)
        else:
            total_points = len(self.all_data) if self.all_data is not None else 0
            
        return (total_points * self.split + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, index):
        """Get a balanced batch of data"""
        if self.all_data is None:
            raise ValueError("Data not loaded")
        
        if self.balance_classes:
            return self._get_balanced_batch(index)
        else:
            return self._get_regular_batch(index)
    
    def _get_balanced_batch(self, index):
        """Get a batch with balanced classes"""
        # Calculate how many samples per class in this batch
        unique_classes = list(self.class_indices.keys()) # Class labels, returns for example: [0, 1]
        samples_per_class = self.batch_size // len(unique_classes) # Ex: 1024 / 2
        remainder = self.batch_size % len(unique_classes) # Just in case
    
        batch_indices = []
        
        # Sample from each class
        for i, class_id in enumerate(unique_classes): # Iterate [0,0] -> [1,1] -> ...
            # class_indices[0] = [...] <- array([0, 1, 2, 5, 8, 10, ...]) with label 0 at these positions
            class_indices = self.class_indices[class_id] 
            
            # Determine how many samples for this class
            num_samples = samples_per_class
            if i < remainder:  
                num_samples += 1 # Distributelleft over sample by giving remainder sample to first calss
            
            if len(class_indices) >= num_samples: # Ex: if number of all point with calss 0 in first iteration > sample per class
                sampled_indices = np.random.choice(class_indices, num_samples, replace=False) # Take randomly num_samples from class_indices and add to batch
            else:
                sampled_indices = np.random.choice(class_indices, num_samples, replace=True) # Just in case
            
            batch_indices.extend(sampled_indices) # Finish balancing
        
        # Shuffle the batch indices
        batch_indices = np.array(batch_indices)
        if self.shuffle:
            np.random.shuffle(batch_indices)
        
        # Get data and labels
        batch_data = self.all_data[batch_indices]
        batch_labels = self.all_labels[batch_indices]
        
        # Format output based on output_format parameter
        if self.output_format == 'dict':
            # Dictionary format for PCEDNet
            inputs = {f'scale_input_{i}': batch_data[:, i, :] for i in range(self.target_scales)}
        elif self.output_format == 'flat':
            # Flattened format for SimpleEdgeNet
            inputs = batch_data.reshape(batch_data.shape[0], -1)
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")
        
        return inputs, batch_labels.reshape(-1, 1)
    
    def _get_regular_batch(self, index):
        """Get a regular batch without balancing"""
        batch_start = index * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.all_data))
        
        # Get batch indices
        if batch_end > len(self.all_data):
            # Wrap around if we exceed data length
            indices = list(range(batch_start, len(self.all_data))) + \
                     list(range(0, batch_end - len(self.all_data)))
        else:
            indices = list(range(batch_start, batch_end))
        
        # Pad if necessary
        while len(indices) < self.batch_size:
            indices.extend(range(min(len(self.all_data), self.batch_size - len(indices))))
        
        indices = indices[:self.batch_size]
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        batch_data = self.all_data[indices]
        batch_labels = self.all_labels[indices]
        
        # Format output based on output_format parameter
        if self.output_format == 'dict':
            # Dictionary format for PCEDNet
            inputs = {f'scale_input_{i}': batch_data[:, i, :] for i in range(self.target_scales)}
        elif self.output_format == 'flat':
            # Flattened format for SimpleEdgeNet
            inputs = batch_data.reshape(batch_data.shape[0], -1)
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")
        
        return inputs, batch_labels.reshape(-1, 1)
    
    def on_epoch_end(self):
        """Reset and shuffle: points pool (with all points with different class) at epoch end"""
        if self.shuffle and self.balance_classes and self.indices_pool is not None:
            np.random.shuffle(self.indices_pool)
        self.current_index = 0