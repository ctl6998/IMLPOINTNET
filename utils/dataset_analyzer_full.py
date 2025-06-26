import os
import numpy as np

def analyze_full_dataset(base_path: str):
    """
    Analyze the entire dataset (all files) for comprehensive statistics
    """
    print(f"\n=== FULL DATASET ANALYSIS ===")
    print(f"Analyzing all files in: {base_path}")
    
    ssm_path = os.path.join(base_path, "SSM_Challenge-ABC")
    lb_path = os.path.join(base_path, "lb")
    
    if not os.path.exists(lb_path):
        print("Label directory not found!")
        return
    
    lb_files = [f for f in os.listdir(lb_path) if f.endswith('.lb')]
    
    if not lb_files:
        print("No label files found!")
        return
    
    print(f"Analyzing {len(lb_files)} label files...")
    
    total_points = 0
    total_non_annotated = 0
    total_non_edge = 0
    total_edge = 0
    
    for lb_file in lb_files:
        lb_file_path = os.path.join(lb_path, lb_file)
        
        try:
            with open(lb_file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse labels
            labels = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    try:
                        labels.append(int(line.strip()))
                    except ValueError:
                        continue
            
            if not labels:
                continue
                
            labels = np.array(labels)
            
            # Count labels
            file_total = len(labels)
            file_non_annotated = np.sum(labels == -1)
            file_non_edge = np.sum(labels == 0)
            file_edge = np.sum(labels == 1)
            
            total_points += file_total
            total_non_annotated += file_non_annotated
            total_non_edge += file_non_edge
            total_edge += file_edge
            
        except Exception as e:
            print(f"Error processing {lb_file}: {e}")
            continue
    
    # Print comprehensive statistics
    annotated_total = total_points - total_non_annotated
    
    print(f"\n=== COMPREHENSIVE DATASET STATISTICS ===")
    print(f"Total files processed: {len(lb_files)}")
    print(f"Total points: {total_points:,}")
    print(f"Non-annotated (-1): {total_non_annotated:,} ({total_non_annotated/total_points*100:.2f}%)")
    print(f"Non-edge (0): {total_non_edge:,} ({total_non_edge/total_points*100:.2f}%)")
    print(f"Edge (1): {total_edge:,} ({total_edge/total_points*100:.2f}%)")
    
    print(f"\n=== TRAINING DATA STATISTICS (annotated only) ===")
    if annotated_total > 0:
        print(f"Annotated points: {annotated_total:,}")
        print(f"Training non-edge: {total_non_edge:,} ({total_non_edge/annotated_total*100:.2f}%)")
        print(f"Training edge: {total_edge:,} ({total_edge/annotated_total*100:.2f}%)")
        print(f"Class imbalance ratio (edge/non-edge): {total_edge/max(total_non_edge, 1):.4f}")
        
        # Calculate how many samples each class needs for perfect balance
        if total_edge != total_non_edge:
            max_class_size = max(total_edge, total_non_edge)
            min_class_size = min(total_edge, total_non_edge)
            imbalance_factor = max_class_size / min_class_size
            print(f"Imbalance factor: {imbalance_factor:.2f}x")
            print(f"For balanced training, minority class needs {max_class_size - min_class_size:,} more samples")
    else:
        print("No annotated points found for training!")
    
    print(f"\n=== DATA FILTERING IMPACT ===")
    print(f"Points excluded from training: {total_non_annotated:,} ({total_non_annotated/total_points*100:.2f}%)")
    print(f"Points used for training: {annotated_total:,} ({annotated_total/total_points*100:.2f}%)")
    print(f"Data utilization efficiency: {annotated_total/total_points*100:.2f}%")