# lasso regression on all 320 features (20 descriptors x 16 scales)
# test is generate from train

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import os
import glob

def load_abc_data(ssm_path, lb_path, max_files=None):
    """Load ABC dataset from .ssm and .lb files
    
    Args:
        ssm_path: Path to SSM files directory
        lb_path: Path to label files directory  
        max_files: Maximum number of files to load (None for all files)
    """
    X_list = []
    y_list = []
    
    ssm_files = sorted(glob.glob(os.path.join(ssm_path, "*.ssm")))
    
    # Limit number of files if specified
    if max_files is not None:
        ssm_files = ssm_files[:max_files]
        print(f"Loading first {len(ssm_files)} files out of available files")
    
    for ssm_file in ssm_files:
        # Get corresponding label file
        base_name = os.path.basename(ssm_file).replace('.ssm', '.lb')
        lb_file = os.path.join(lb_path, base_name)
        
        if os.path.exists(lb_file):
            # Load SSM features with proper header parsing
            with open(ssm_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            n_points, total_features = map(int, lines[0].split())
            features_per_scale = int(lines[1])
            n_scales = int(lines[3])
            
            # Load feature data (skip header lines)
            data_lines = lines[5:]
            features = []
            
            for line in data_lines:
                if line.strip():
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) == total_features:  # 320 features
                            features.append(values)
                    except ValueError:
                        continue
            
            features = np.array(features)
            
            # Load labels (skip first line which is header)
            with open(lb_file, 'r') as f:
                lines = f.readlines()
            
            labels = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    try:
                        labels.append(int(line.strip()))
                    except ValueError:
                        continue
            
            labels = np.array(labels)
            
            # Ensure same length
            min_length = min(len(features), len(labels))
            features = features[:min_length]
            labels = labels[:min_length]
            
            # Filter out non-annotated points (label = -1)
            annotated_mask = (labels != -1)
            features = features[annotated_mask]
            labels = labels[annotated_mask]
            
            if len(features) > 0:
                X_list.append(features)
                y_list.append(labels)
                print(f"Loaded {len(features)} points from {os.path.basename(ssm_file)}")
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    return X, y

# Load actual ABC data
ssm_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/SSM_Challenge-ABC"
lb_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/lb"

print("Loading ABC dataset...")
X, y = load_abc_data(ssm_path, lb_path, max_files=5)  #Load only first 5 files
print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply LassoCV for feature selection
print("Applying LassoCV for feature selection...")
start_time = time.time()

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=1000) #Cross-validation with 5 folds
lasso_cv.fit(X_train_scaled, y_train)

fitting_time = time.time() - start_time

# Get results
optimal_alpha = lasso_cv.alpha_
coefficients = lasso_cv.coef_
selected_features = np.where(coefficients != 0)[0]
n_selected = len(selected_features)
percentage_selected = (n_selected / 320) * 100

# Print results
print(f"\nFitting time: {fitting_time:.3f} seconds")
print(f"Optimal alpha (regularization strength): {optimal_alpha:.6f}")
print(f"Selected features: {n_selected} out of 320")
print(f"Percentage selected: {percentage_selected:.1f}%")

# Reshape coefficients to (20 descriptors, 16 scales)
coef_matrix = coefficients.reshape(20, 16)

# Calculate descriptor importance (sum across scales)
descriptor_importance = np.abs(coef_matrix).sum(axis=1)
descriptor_selected_scales = (coef_matrix != 0).sum(axis=1)

# Print descriptor analysis
print("\nDescriptor Analysis (sorted by total importance):")
print("Descriptor | Total Weight | Selected Scales")
print("-" * 45)
for i in np.argsort(descriptor_importance)[::-1]:
    print(f"D{i+1:2d}       | {descriptor_importance[i]:10.4f} | {descriptor_selected_scales[i]:2d}/16")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Heatmap of coefficients (descriptors vs scales)
im1 = axes[0,0].imshow(np.abs(coef_matrix), cmap='viridis', aspect='auto')
axes[0,0].set_title('Feature Selection Heatmap\n(Absolute Lasso Coefficients)')
axes[0,0].set_xlabel('Scale Index (1-16)')
axes[0,0].set_ylabel('Descriptor Index (1-20)')
axes[0,0].set_xticks(range(0, 16, 2))
axes[0,0].set_xticklabels(range(1, 17, 2))
axes[0,0].set_yticks(range(0, 20, 2))
axes[0,0].set_yticklabels(range(1, 21, 2))
plt.colorbar(im1, ax=axes[0,0], label='|Coefficient|')

# 2. Column graph of descriptor total weights
descriptor_indices = np.arange(1, 21)
bars = axes[0,1].bar(descriptor_indices, descriptor_importance)
axes[0,1].set_title('Total Descriptor Importance\n(Sum of |Coefficients| across scales)')
axes[0,1].set_xlabel('Descriptor Index')
axes[0,1].set_ylabel('Total Weight')
axes[0,1].set_xticks(range(1, 21, 2))
axes[0,1].grid(True, alpha=0.3)

# Color bars based on importance
max_importance = max(descriptor_importance)
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.viridis(descriptor_importance[i] / max_importance))

# 3. Number of selected scales per descriptor
axes[1,0].bar(descriptor_indices, descriptor_selected_scales, color='orange', alpha=0.7)
axes[1,0].set_title('Number of Selected Scales per Descriptor')
axes[1,0].set_xlabel('Descriptor Index')
axes[1,0].set_ylabel('Number of Selected Scales')
axes[1,0].set_xticks(range(1, 21, 2))
axes[1,0].set_ylim(0, 16)
axes[1,0].grid(True, alpha=0.3)

# 4. Feature selection pattern across scales
scale_selection_count = (coef_matrix != 0).sum(axis=0)
axes[1,1].bar(range(1, 17), scale_selection_count, color='green', alpha=0.7)
axes[1,1].set_title('Feature Selection Count per Scale')
axes[1,1].set_xlabel('Scale Index')
axes[1,1].set_ylabel('Number of Selected Descriptors')
axes[1,1].set_xticks(range(1, 17, 2))
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nSummary:")
print(f"- Fitting completed in {fitting_time:.3f} seconds")
print(f"- Optimal regularization strength (alpha): {optimal_alpha:.6f}")
print(f"- Selected {n_selected} features out of 320 ({percentage_selected:.1f}%)")
print(f"- Most important descriptor: D{np.argmax(descriptor_importance)+1}")
print(f"- Descriptor with most selected scales: D{np.argmax(descriptor_selected_scales)+1} ({max(descriptor_selected_scales)} scales)")