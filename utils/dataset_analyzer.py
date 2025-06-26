import os
import numpy as np

def analyze_dataset(base_path: str):
    """
    Analyze the dataset structure and statistics
    """
    print(f"Analyzing dataset at: {base_path}")
    ssm_path = os.path.join(base_path, "SSM_Challenge-ABC")
    lb_path = os.path.join(base_path, "lb")
    ply_path = os.path.join(base_path, "ply")
    
    ssm_files = len([f for f in os.listdir(ssm_path) if f.endswith('.ssm')]) if os.path.exists(ssm_path) else 0
    lb_files = len([f for f in os.listdir(lb_path) if f.endswith('.lb')]) if os.path.exists(lb_path) else 0
    ply_files = len([f for f in os.listdir(ply_path) if f.endswith('.ply')]) if os.path.exists(ply_path) else 0
    
    print(f"SSM files: {ssm_files}")
    print(f"LB files: {lb_files}")
    print(f"PLY files: {ply_files}")

    if ssm_files > 0:
        sample_ssm = [f for f in os.listdir(ssm_path) if f.endswith('.ssm')][0]
        sample_ssm_path = os.path.join(ssm_path, sample_ssm)
        
        try:
            with open(sample_ssm_path, 'r') as f:
                lines = f.readlines()
            
            n_points, n_features = map(int, lines[0].split())
            n_scales = int(lines[1])
            features_per_scale = n_features // n_scales
            
            print(f"Sample SSM file analysis ({sample_ssm}):")
            print(f"  Points: {n_points}")
            print(f"  Total features: {n_features}")
            print(f"  Scales: {n_scales}")
            print(f"  Features per scale: {features_per_scale}")
            
        except Exception as e:
            print(f"Error analyzing sample file: {e}")
    
    if lb_files > 0:
        sample_lb = [f for f in os.listdir(lb_path) if f.endswith('.lb')][0]
        sample_lb_path = os.path.join(lb_path, sample_lb)
        
        try:
            with open(sample_lb_path, 'r') as f:
                lines = f.readlines()
            
            # Parse all labels, handling the new format
            labels = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    try:
                        labels.append(int(line.strip()))
                    except ValueError:
                        continue
            
            labels = np.array(labels)
            
            # Count different label types with new logic
            non_annotated_count = np.sum(labels == -1)
            non_edge_count = np.sum(labels == 0)  # Class 0: non-edge
            edge_count = np.sum(labels == 1)      # Class 1: edge
            
            total_points = len(labels)
            annotated_points = total_points - non_annotated_count
            
            print(f"Sample label file analysis ({sample_lb}):")
            print(f"  Total points: {total_points}")
            print(f"  Non-annotated (-1): {non_annotated_count} ({non_annotated_count/total_points*100:.2f}%)")
            print(f"  Non-edge (0): {non_edge_count} ({non_edge_count/total_points*100:.2f}%)")
            print(f"  Edge (1): {edge_count} ({edge_count/total_points*100:.2f}%)")
            print(f"  --- Training Data (annotated only) ---")
            if annotated_points > 0:
                print(f"  Annotated points: {annotated_points}")
                print(f"  Training non-edge: {non_edge_count} ({non_edge_count/annotated_points*100:.2f}%)")
                print(f"  Training edge: {edge_count} ({edge_count/annotated_points*100:.2f}%)")
                print(f"  Class balance ratio (edge/non-edge): {edge_count/max(non_edge_count, 1):.3f}")
            else:
                print(f"  No annotated points found for training!")
                
        except Exception as e:
            print(f"Error analyzing sample labels: {e}")

