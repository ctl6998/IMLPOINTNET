# ========================================
# 0. Import Libraries and Setup
# ========================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os
import glob

# Fix for Python 3.10+ compatibility with STG library
import collections
import collections.abc
collections.Sequence = collections.abc.Sequence
collections.Set = collections.abc.Set
collections.Mapping = collections.abc.Mapping

from stg import STG

# ========================================
# 1. Load ABC Dataset
# ========================================

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

print("Loading ABC dataset...")
ssm_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/SSM_Challenge-ABC"
lb_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/lb"
X, y = load_abc_data(ssm_path, lb_path, max_files=19) #Fraction = 0.1

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ========================================
# 2. Data Preprocessing and Splitting
# ========================================

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ========================================
# 3. STG Model Setup and Training
# ========================================

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create STG model
model = STG(
    task_type='classification',
    input_dim=X_train.shape[1],
    output_dim=2,
    hidden_dims=[100, 200, 320, 400],  # mimic Lassonet hidden layer d/3,2d/3,d,4d/3
    activation='relu',
    optimizer='Adam',
    learning_rate=0.001,
    batch_size=512,
    feature_selection=True,
    sigma=0.5,
    lam=0.1,  # higher = more feature zero-out
    random_state=42,
    device=device
)

# Train model
print("Training STG model...")
model.fit(
    X_train, y_train,
    nr_epochs= 200,  # 500,100
    valid_X=X_val,
    valid_y=y_val,
    print_interval=100
)

# ========================================
# 4. Feature Importance Analysis
# ========================================

# Get feature importance
gates_prob = model.get_gates(mode='prob')  # Feature weight but shown in probability
gates_raw = model.get_gates(mode='raw')
print(f"Gate probabilities shape: {gates_prob.shape}")

# Reshape to 20 descriptors x 16 scales
n_descriptors = 20
n_scales = 16
gates_matrix = gates_prob.reshape(n_descriptors, n_scales)

# Calculate descriptor weights (sum across scales)
descriptor_weights = np.sum(gates_matrix, axis=1)

# ========================================
# 5. Results Analysis and Printing
# ========================================

print("\n" + "="*50)
print("FEATURE SELECTION RESULTS")
print("="*50)

# Complete descriptor ranking - print ALL descriptors ranked by weight
print("\nCOMPLETE DESCRIPTOR RANKING BY TOTAL WEIGHT:")
print("-" * 50)
ranked_descriptors = np.argsort(descriptor_weights)[::-1]
for rank, desc_idx in enumerate(ranked_descriptors):
    print(f"Rank {rank+1:2d}: Descriptor {desc_idx+1:2d} - Weight = {descriptor_weights[desc_idx]:.4f}")

# Scale importance
scale_weights = np.sum(gates_matrix, axis=0)
top_scales = np.argsort(scale_weights)[::-1]
print("\nTop 5 Most Important Scales:")
for i, scale_idx in enumerate(top_scales[:5]):
    print(f"  {i+1}. Scale {scale_idx+1}: Weight = {scale_weights[scale_idx]:.4f}")

# Feature selection summary
threshold = 0.5
selected_features = np.sum(gates_matrix > threshold)
print(f"\nSelected features (prob > {threshold}): {selected_features}/{n_descriptors*n_scales}")
print(f"Selection ratio: {selected_features/(n_descriptors*n_scales):.2%}")

# Print selected features by descriptor and scale
selected_indices = np.where(gates_matrix > threshold)
if len(selected_indices[0]) > 0:
    print(f"\nSelected Feature Combinations (prob > {threshold}):")
    for i, (desc_idx, scale_idx) in enumerate(zip(selected_indices[0], selected_indices[1])):
        prob = gates_matrix[desc_idx, scale_idx]
        print(f"  {i+1}. Descriptor {desc_idx+1} - Scale {scale_idx+1}: {prob:.4f}")
else:
    print(f"\nNo features selected with threshold > {threshold}")
    # Lower threshold to show some features
    lower_threshold = 0.3
    selected_indices_lower = np.where(gates_matrix > lower_threshold)
    if len(selected_indices_lower[0]) > 0:
        print(f"\nFeatures with prob > {lower_threshold}:")
        for i, (desc_idx, scale_idx) in enumerate(zip(selected_indices_lower[0], selected_indices_lower[1])):
            prob = gates_matrix[desc_idx, scale_idx]
            print(f"  {i+1}. Descriptor {desc_idx+1} - Scale {scale_idx+1}: {prob:.4f}")
            if i >= 9:  # Limit to top 10
                break

# ========================================
# 6. Model Evaluation
# ========================================

# Evaluate model - fix device mismatch and key name
X_test_tensor = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    model._model.eval()
    feed_dict = {'input': X_test_tensor}
    output = model._model(feed_dict)
    y_pred = torch.argmax(output['logits'], dim=1).cpu().numpy()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ========================================
# 7. Visualizations
# ========================================

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Heatmap of descriptors x scales
plt.subplot(2, 2, 1)
plt.imshow(gates_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Gate Probability')
plt.title('Feature Importance: Descriptors × Scales')
plt.xlabel('Scale')
plt.ylabel('Descriptor')
plt.xticks(range(n_scales), range(1, n_scales+1))
plt.yticks(range(n_descriptors), range(1, n_descriptors+1))

# 2. Descriptor weights bar plot with 4 decimal places
plt.subplot(2, 2, 2)
bars = plt.bar(range(1, n_descriptors+1), descriptor_weights, color='steelblue', alpha=0.7)
plt.bar_label(bars, fmt='%.4f')  # Changed to 4 decimal places
plt.title('Total Weight per Descriptor')
plt.xlabel('Descriptor ID')
plt.ylabel('Total Weight (across 16 scales)')
plt.xticks(range(1, n_descriptors+1))
plt.grid(True, alpha=0.3)
# Rotate x-axis labels to prevent overlap
plt.xticks(rotation=45)

# 3. Scale weights bar plot
plt.subplot(2, 2, 3)
scale_bars = plt.bar(range(1, n_scales+1), scale_weights, color='orange', alpha=0.7)
plt.bar_label(scale_bars, fmt='%.4f')  # Also added 4 decimal places for consistency
plt.title('Total Weight per Scale')
plt.xlabel('Scale ID')
plt.ylabel('Total Weight (across 20 descriptors)')
plt.xticks(range(1, n_scales+1))
plt.grid(True, alpha=0.3)

# 4. Top feature combinations
plt.subplot(2, 2, 4)
flat_indices = np.argsort(gates_matrix.flatten())[::-1][:10]
top_combinations = []
top_values = []
for flat_idx in flat_indices:
    desc_idx, scale_idx = np.unravel_index(flat_idx, gates_matrix.shape)
    top_combinations.append(f'D{desc_idx+1}S{scale_idx+1}')
    top_values.append(gates_matrix[desc_idx, scale_idx])

bars_horizontal = plt.barh(range(len(top_combinations)), top_values, color='green', alpha=0.7)
# Add value labels to horizontal bars
for i, (bar, value) in enumerate(zip(bars_horizontal, top_values)):
    plt.text(value + 0.001, i, f'{value:.4f}', va='center', ha='left', fontsize=8)
plt.yticks(range(len(top_combinations)), top_combinations)
plt.xlabel('Gate Probability')
plt.title('Top 10 Feature Combinations')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stg_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = np.corrcoef(gates_matrix)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Descriptor Correlation Matrix')
plt.xlabel('Descriptor ID')
plt.ylabel('Descriptor ID')
descriptor_labels = [f'D{i+1}' for i in range(n_descriptors)]
plt.xticks(range(n_descriptors), descriptor_labels, rotation=45)
plt.yticks(range(n_descriptors), descriptor_labels)
plt.tight_layout()
plt.savefig('stg_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("STG feature selection completed!")