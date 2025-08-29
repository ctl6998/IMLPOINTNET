# import numpy as np
# import pandas as pd
# import glob
# import os
# import time
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Install LassoNet if not already installed
# # pip install lassonet

# from lassonet import LassoNetClassifierCV, LassoNetClassifier, plot_path, plot_cv

# # ========================================
# # 1. Load ABC Dataset
# # ========================================

# def load_abc_data(ssm_path, lb_path, max_files=None):
#     """Load ABC dataset from .ssm and .lb files
    
#     Args:
#         ssm_path: Path to SSM files directory
#         lb_path: Path to label files directory  
#         max_files: Maximum number of files to load (None for all files)
#     """
#     X_list = []
#     y_list = []
    
#     ssm_files = sorted(glob.glob(os.path.join(ssm_path, "*.ssm")))
    
#     # Limit number of files if specified
#     if max_files is not None:
#         ssm_files = ssm_files[:max_files]
#         print(f"Loading first {len(ssm_files)} files out of available files")
    
#     for ssm_file in ssm_files:
#         # Get corresponding label file
#         base_name = os.path.basename(ssm_file).replace('.ssm', '.lb')
#         lb_file = os.path.join(lb_path, base_name)
        
#         if os.path.exists(lb_file):
#             # Load SSM features with proper header parsing
#             with open(ssm_file, 'r') as f:
#                 lines = f.readlines()
            
#             # Parse header
#             n_points, total_features = map(int, lines[0].split())
#             features_per_scale = int(lines[1])
#             n_scales = int(lines[3])
            
#             # Load feature data (skip header lines)
#             data_lines = lines[5:]
#             features = []
            
#             for line in data_lines:
#                 if line.strip():
#                     try:
#                         values = [float(x) for x in line.split()]
#                         if len(values) == total_features:  # 320 features
#                             features.append(values)
#                     except ValueError:
#                         continue
            
#             features = np.array(features)
            
#             # Load labels (skip first line which is header)
#             with open(lb_file, 'r') as f:
#                 lines = f.readlines()
            
#             labels = []
#             for line in lines[1:]:  # Skip header
#                 if line.strip():
#                     try:
#                         labels.append(int(line.strip()))
#                     except ValueError:
#                         continue
            
#             labels = np.array(labels)
            
#             # Ensure same length
#             min_length = min(len(features), len(labels))
#             features = features[:min_length]
#             labels = labels[:min_length]
            
#             # Filter out non-annotated points (label = -1)
#             annotated_mask = (labels != -1)
#             features = features[annotated_mask]
#             labels = labels[annotated_mask]
            
#             if len(features) > 0:
#                 X_list.append(features)
#                 y_list.append(labels)
#                 print(f"Loaded {len(features)} points from {os.path.basename(ssm_file)}")
    
#     X = np.vstack(X_list)
#     y = np.hstack(y_list)
    
#     return X, y

# # ABC Dataset structure helper functions
# def get_descriptor_info(feature_idx, features_per_scale=20, n_scales=16):
#     """Get descriptor type and scale for a given feature index"""
#     # For ABC dataset: 320 features = 20 descriptors × 16 scales
#     # Features are organized as: [desc0_scale0, desc0_scale1, ..., desc0_scale15, desc1_scale0, ...]
#     descriptor_idx = feature_idx // n_scales
#     scale_idx = feature_idx % n_scales
    
#     # Ensure indices are within bounds
#     descriptor_idx = min(descriptor_idx, features_per_scale - 1)
#     scale_idx = min(scale_idx, n_scales - 1)
    
#     return descriptor_idx, scale_idx

# def analyze_selected_descriptors(selected_features, features_per_scale=20, n_scales=16, feature_importances=None):
#     """Analyze which descriptors and scales were selected"""
#     # Descriptor types for ABC dataset (20 descriptors × 16 scales = 320 features)
#     descriptor_names = [
#         'd1', 'd2', 'd3', 'd4',
#         'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13',
#         'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20'
#     ]
    
#     # Initialize tracking arrays
#     descriptor_weights = np.zeros(features_per_scale)
#     descriptor_counts = np.zeros(features_per_scale)
#     scale_matrix = np.zeros((features_per_scale, n_scales))
    
#     # Convert feature_importances to numpy array if it's a tensor
#     if feature_importances is not None:
#         if hasattr(feature_importances, 'cpu'):  # PyTorch tensor
#             feature_importances = feature_importances.cpu().numpy()
#         elif hasattr(feature_importances, 'numpy'):  # TensorFlow tensor
#             feature_importances = feature_importances.numpy()
#         else:
#             feature_importances = np.array(feature_importances)
    
#     for feat_idx in selected_features:
#         descriptor_idx, scale_idx = get_descriptor_info(feat_idx, features_per_scale, n_scales)
        
#         # Count selections
#         descriptor_counts[descriptor_idx] += 1
#         scale_matrix[descriptor_idx, scale_idx] = 1
        
#         # Add importance weights if available
#         if feature_importances is not None:
#             descriptor_weights[descriptor_idx] += float(feature_importances[feat_idx])
#         else:
#             descriptor_weights[descriptor_idx] += 1
    
#     return descriptor_names, descriptor_weights, descriptor_counts, scale_matrix

# # Load actual ABC data
# ssm_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/SSM_Challenge-ABC"
# lb_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/lb"

# # Start with a subset for testing, then increase
# print("Loading ABC Dataset...")
# X, y = load_abc_data(ssm_path, lb_path, max_files=19)  # Start with 5 files
# print(f"Dataset shape: {X.shape}")
# print(f"Classes distribution: {np.bincount(y)}")

# # ========================================
# # 2. Data Preprocessing
# # ========================================

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # IMPORTANT: Normalize the data (LassoNet requires this)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# print(f"Training set shape: {X_train_scaled.shape}")
# print(f"Test set shape: {X_test_scaled.shape}")

# # ========================================
# # 3. Method 1: Cross-Validation Interface - devide training into groups/fold, each group folds try different alpha. Receive alpha for the best accuracy on test set
# # Problem, LassoNet use hidden layer to check the final classified accuracy. 
# # For example, increaseing alpha == zero out some features == higher accuarcy, this motivated the cross-validation to choose this alpha.
# # But incase of ABC, (with a simple hidden layer) fully selected features == max accuracy ---> zero out some features == always smaller accuarcy. Remember that the goal is not accuracy, it is the speed!
# # In case of ABC, it always have max performance on 20x16. The problem is that we choose less to optimize performance. 
# # LassoNetCV really don't have motivation to select less (if the motivation is accuracy).
# # Question 1? Architecture for ABC to reach higher accuracy with less featutred/descriptor seletecd?
# # Question 2? Can we modify LassoNet so it take into account the speed too? Neural network with time-loss (performance) not accuracy(label - predict)? This doesn't make anysense, less featreus ~ better performance, beside we don't have such thing called "label-training time"!
# # ========================================

# # Add diagnostic information before training
# print("\n" + "="*60)
# print("DIAGNOSTIC INFORMATION")
# print("="*60)
# print(f"Data shape: {X.shape}")
# print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
# print(f"Data mean: {X.mean():.3f}, std: {X.std():.3f}")
# print(f"After scaling - mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")
# print(f"Class distribution: {np.bincount(y)}")
# print(f"Class percentages: {np.bincount(y) / len(y) * 100}")

# # Check if dataset is too easy (perfect separability)
# from sklearn.linear_model import LogisticRegression
# simple_model = LogisticRegression(max_iter=1000)
# simple_model.fit(X_train_scaled, y_train)
# simple_accuracy = simple_model.score(X_test_scaled, y_test)
# print(f"Simple Logistic Regression accuracy: {simple_accuracy:.4f}")

# if simple_accuracy > 0.98:
#     print("⚠️  WARNING: Dataset appears to be linearly separable!")
#     print("   This may explain why feature selection doesn't occur.")
#     print("   Consider using a more challenging subset or different classes.")

# print("\n" + "="*60)
# print("Method 1: Cross-Validation Approach")
# print("="*60)

# # Track fitting time
# start_time = time.time()

# # Use LassoNetClassifierCV for automatic hyperparameter selection
# model_cv = LassoNetClassifierCV(
#     hidden_dims=(100, 200, 320, 400),      # 2d/3 for d=320 as per paper recommendation, this mean layer: 320x1 -> 213x1 -> 320x1 -> num_classesx1
#     path_multiplier=1.05,    # Controls regularization path granularity
#     M=10,                    # Hierarchy parameter (default works well)
#     cv=5,                    # 5-fold cross-validation, devide the training set into 5 and train, result = average of those 5. Each have auto-assigned lambda-start. The more the better the slower.
#     lambda_start="auto",     # auto=0
#     random_state=42
# )

# # Fit the model (this will automatically find the best lambda)
# print("Training LassoNet with cross-validation...")
# path_cv = model_cv.fit(X_train_scaled, y_train)

# fitting_time_cv = time.time() - start_time

# # Evaluate the model
# y_pred_cv = model_cv.predict(X_test_scaled)
# accuracy_cv = accuracy_score(y_test, y_pred_cv)

# # Results Summary for Method 1
# print(f"\n{'='*40}")
# print("METHOD 1 RESULTS (Cross-Validation)")
# print(f"{'='*40}")
# print(f"Fitting time: {fitting_time_cv:.2f} seconds")
# print(f"Optimal alpha (lambda): {model_cv.best_lambda_:.6f}")
# print(f"Selected features: {len(model_cv.best_selected_)} out of {X.shape[1]}")
# print(f"Feature selection percentage: {len(model_cv.best_selected_)/X.shape[1]*100:.1f}%")
# print(f"Best CV score: {model_cv.best_cv_score_:.4f}")
# print(f"Test accuracy: {accuracy_cv:.4f}")

# # Check if feature selection actually occurred
# if len(model_cv.best_selected_) == X.shape[1]:
#     print("⚠️  WARNING: No feature selection occurred! All features were selected.")
#     print("   This suggests lambda is too small. Consider:")
#     print("   - Using a larger lambda_start")
#     print("   - Increasing path_multiplier")
#     print("   - Check if data scaling is appropriate")

# # Analyze selected descriptors for Method 1
# try:
#     descriptor_names, descriptor_weights_cv, descriptor_counts_cv, scale_matrix_cv = analyze_selected_descriptors(
#         model_cv.best_selected_, feature_importances=model_cv.feature_importances_
#     )
    
#     print(f"\nDescriptor Analysis (Total Importance/Weight):")
#     print("-" * 60)
#     sorted_indices = np.argsort(descriptor_weights_cv)[::-1]
#     for i, desc_idx in enumerate(sorted_indices):
#         if descriptor_weights_cv[desc_idx] > 0:
#             n_scales = int(descriptor_counts_cv[desc_idx])
#             weight = descriptor_weights_cv[desc_idx]
#             print(f"{i+1:2d}. {descriptor_names[desc_idx]:20s} | Weight: {weight:8.4f} | Scales: {n_scales:2d}/16")
            
#             # Only show top 15 to avoid too much output
#             if i >= 14:
#                 remaining = np.sum(descriptor_weights_cv > 0) - 15
#                 if remaining > 0:
#                     print(f"     ... and {remaining} more descriptors")
#                 break
                
# except Exception as e:
#     print(f"Error in descriptor analysis: {e}")
#     print("Continuing with basic analysis...")
#     descriptor_weights_cv = np.ones(20)  # Fallback
#     descriptor_counts_cv = np.ones(20)
#     scale_matrix_cv = np.ones((20, 16))

# # ========================================
# # 4. Method 2: Manual Path Exploration - choose alpha according to path
# # ========================================

# print("\n" + "="*60)
# print("Method 2: Manual Path Exploration")
# print("="*60)


# start_time = time.time()

# # Model
# model_base = LassoNetClassifier(
#     hidden_dims=(100, 200, 320, 400),
#     M=10,
#     random_state=42
# )

# print("Computing regularization path...")
# #Create path with default lambda_start and path_muttiplier (default when we generate the LassoNetClassifier)
# #Path λ (regularization strength) = [lamda_start=0.0(default), 0.01, 0.05,.... 5.0]. With each λ thenetwork reducate/unconnected a set of features until it uncoonected all!
# path = model_base.path(X_train_scaled, y_train, return_state_dicts=True) 

# fitting_time_base = time.time() - start_time

# # Find best model from path based on validation score
# best_idx = 0
# best_score = 0
# for i, checkpoint in enumerate(path):
#     model_base.load(checkpoint.state_dict)
#     score = model_base.score(X_test_scaled, y_test)
#     if score > best_score:
#         best_score = score
#         best_idx = i

# # Load best model
# model_base.load(path[best_idx].state_dict)
# best_lambda = path[best_idx].lambda_
# selected_features_base = path[best_idx].selected

# # Results Summary for Method 2
# print(f"\n{'='*40}")
# print("METHOD 2 RESULTS (Manual Path)")
# print(f"{'='*40}")
# print(f"Fitting time: {fitting_time_base:.2f} seconds")
# print(f"Optimal alpha (lambda): {best_lambda:.6f}")
# print(f"Selected features: {len(selected_features_base)} out of {X.shape[1]}")
# print(f"Feature selection percentage: {len(selected_features_base)/X.shape[1]*100:.1f}%")
# print(f"Test accuracy: {best_score:.4f}")

# # Analyze selected descriptors for Method 2
# try:
#     descriptor_names, descriptor_weights_base, descriptor_counts_base, scale_matrix_base = analyze_selected_descriptors(
#         selected_features_base
#     )

#     print(f"\nDescriptor Analysis (Selection Count):")
#     print("-" * 60)
#     sorted_indices = np.argsort(descriptor_counts_base)[::-1]
#     for i, desc_idx in enumerate(sorted_indices):
#         if descriptor_counts_base[desc_idx] > 0:
#             n_scales = int(descriptor_counts_base[desc_idx])
#             print(f"{i+1:2d}. {descriptor_names[desc_idx]:20s} | Count: {n_scales:2d}/16 scales")
            
#             # Only show top 15 to avoid too much output
#             if i >= 14:
#                 remaining = np.sum(descriptor_counts_base > 0) - 15
#                 if remaining > 0:
#                     print(f"     ... and {remaining} more descriptors")
#                 break
# except Exception as e:
#     print(f"Error in descriptor analysis: {e}")
#     print("Continuing with basic analysis...")
#     descriptor_weights_base = np.ones(20)  # Fallback
#     descriptor_counts_base = np.ones(20)
#     scale_matrix_base = np.ones((20, 16))

# # ========================================
# # 5. Visualization
# # ========================================

# print("\n" + "="*60)
# print("Visualization")
# print("="*60)

# # Create comprehensive visualization
# fig = plt.figure(figsize=(20, 12))

# # 1. Cross-validation plot (with error handling)
# plt.subplot(2, 4, 1)
# try:
#     if hasattr(model_cv, 'cv_scores_') and model_cv.cv_scores_ is not None:
#         plot_cv(model_cv, X_test_scaled, y_test)
#         plt.title("CV Results (Method 1)")
#     else:
#         plt.text(0.5, 0.5, 'CV plot not available\n(no feature selection)', 
#                 ha='center', va='center', transform=plt.gca().transAxes)
#         plt.title("CV Results (Method 1) - N/A")
# except Exception as e:
#     plt.text(0.5, 0.5, f'CV plot error:\n{str(e)[:50]}...', 
#             ha='center', va='center', transform=plt.gca().transAxes)
#     plt.title("CV Results (Method 1) - Error")

# # 2. Regularization path plot (with error handling)
# plt.subplot(2, 4, 2)
# try:
#     if len(path) > 1 and any(len(checkpoint.selected) < X.shape[1] for checkpoint in path):
#         plot_path(model_base, X_test_scaled, y_test)
#         plt.title("Regularization Path (Method 2)")
#     else:
#         plt.text(0.5, 0.5, 'Path plot not available\n(no feature selection)', 
#                 ha='center', va='center', transform=plt.gca().transAxes)
#         plt.title("Regularization Path (Method 2) - N/A")
# except Exception as e:
#     plt.text(0.5, 0.5, f'Path plot error:\n{str(e)[:50]}...', 
#             ha='center', va='center', transform=plt.gca().transAxes)
#     plt.title("Regularization Path (Method 2) - Error")

# # 3. Heatmap of selected descriptors/scales (Method 1)
# plt.subplot(2, 4, 3)
# sns.heatmap(scale_matrix_cv, 
#             xticklabels=range(1, 17), 
#             yticklabels=[name[:10] for name in descriptor_names],
#             cmap='Reds', 
#             cbar_kws={'label': 'Selected'})
# plt.title("Selected Features Heatmap\n(Method 1: CV)")
# plt.xlabel("Scale Index")
# plt.ylabel("Descriptor")

# # 4. Column graph of descriptor weights (Method 1)
# plt.subplot(2, 4, 4)
# sorted_indices = np.argsort(descriptor_weights_cv)[::-1]
# top_10_indices = sorted_indices[:10]
# top_10_weights = descriptor_weights_cv[top_10_indices]
# top_10_names = [descriptor_names[i][:10] for i in top_10_indices]

# plt.bar(range(len(top_10_weights)), top_10_weights)
# plt.xticks(range(len(top_10_names)), top_10_names, rotation=45)
# plt.title("Top 10 Descriptor Weights\n(Method 1: CV)")
# plt.ylabel("Total Weight")

# # 5. Heatmap of selected descriptors/scales (Method 2)
# plt.subplot(2, 4, 7)
# sns.heatmap(scale_matrix_base, 
#             xticklabels=range(1, 17), 
#             yticklabels=[name[:10] for name in descriptor_names],
#             cmap='Blues', 
#             cbar_kws={'label': 'Selected'})
# plt.title("Selected Features Heatmap\n(Method 2: Manual)")
# plt.xlabel("Scale Index")
# plt.ylabel("Descriptor")

# # 6. Column graph of descriptor counts (Method 2)
# plt.subplot(2, 4, 8)
# sorted_indices = np.argsort(descriptor_counts_base)[::-1]
# top_10_indices = sorted_indices[:10]
# top_10_counts = descriptor_counts_base[top_10_indices]
# top_10_names = [descriptor_names[i][:10] for i in top_10_indices]

# plt.bar(range(len(top_10_counts)), top_10_counts)
# plt.xticks(range(len(top_10_names)), top_10_names, rotation=45)
# plt.title("Top 10 Descriptor Counts\n(Method 2: Manual)")
# plt.ylabel("Number of Scales Selected")

# # 7. Feature importance comparison
# plt.subplot(2, 4, 5)
# try:
#     if hasattr(model_cv, 'feature_importances_') and model_cv.feature_importances_ is not None:
#         importances = model_cv.feature_importances_
#         if hasattr(importances, 'cpu'):  # Handle tensor
#             importances = importances.cpu().numpy()
#         plt.plot(sorted(importances), 'o-', alpha=0.7)
#         plt.title("Feature Importance Distribution\n(Method 1)")
#         plt.xlabel("Feature Rank")
#         plt.ylabel("Importance")
#     else:
#         plt.text(0.5, 0.5, 'Feature importance\nnot available', 
#                 ha='center', va='center', transform=plt.gca().transAxes)
#         plt.title("Feature Importance - N/A")
# except Exception as e:
#     plt.text(0.5, 0.5, f'Importance error:\n{str(e)[:30]}...', 
#             ha='center', va='center', transform=plt.gca().transAxes)
#     plt.title("Feature Importance - Error")

# # 8. Model comparison
# plt.subplot(2, 4, 6)
# methods = ['CV', 'Manual']
# accuracies = [accuracy_cv, best_score]
# times = [fitting_time_cv, fitting_time_base]
# selected_counts = [len(model_cv.best_selected_), len(selected_features_base)]

# x = np.arange(len(methods))
# width = 0.25

# plt.bar(x - width, accuracies, width, label='Accuracy', alpha=0.7)
# plt.bar(x, np.array(times)/max(times), width, label='Relative Time', alpha=0.7)
# plt.bar(x + width, np.array(selected_counts)/X.shape[1], width, label='Feature %', alpha=0.7)

# plt.xlabel('Method')
# plt.ylabel('Normalized Score')
# plt.title('Method Comparison')
# plt.xticks(x, methods)
# plt.legend()

# plt.tight_layout()
# plt.savefig('lassonet_feature_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()

# # ========================================
# # 7. Train Dense Model on Selected Features
# # ========================================

# print("\n" + "="*60)
# print("Dense Model on Selected Features")
# print("="*60)

# # Use the better performing method
# best_model = model_cv if accuracy_cv >= best_score else model_base
# best_selected = model_cv.best_selected_ if accuracy_cv >= best_score else selected_features_base
# best_method = "CV" if accuracy_cv >= best_score else "Manual"

# print(f"Using {best_method} method results for dense training")

# # Train a dense neural network on only the selected features
# X_train_selected = X_train_scaled[:, best_selected]
# X_test_selected = X_test_scaled[:, best_selected]

# # Train dense model
# start_time = time.time()
# model_dense = LassoNetClassifier(hidden_dims=(100,), M=10)
# model_dense.fit(X_train_selected, y_train, dense_only=True)
# dense_fitting_time = time.time() - start_time

# # Evaluate dense model
# y_pred_dense = model_dense.predict(X_test_selected)
# accuracy_dense = accuracy_score(y_test, y_pred_dense)

# print(f"Dense model training time: {dense_fitting_time:.2f} seconds")
# print(f"Dense model accuracy on {len(best_selected)} selected features: {accuracy_dense:.4f}")
# print(f"Improvement over LassoNet: {accuracy_dense - max(accuracy_cv, best_score):.4f}")

# # ========================================
# # 8. Compare Different Hidden Layer Sizes
# # ========================================

# print("\n" + "="*60)
# print("Comparing Different Architectures")
# print("="*60)

# # Test different architectures as suggested in the paper
# d = X.shape[1]  # Number of features = 320
# architectures = [
#     (d//3,),        # Single layer, d/3 neurons (~107)
#     (d//2,),        # Single layer, d/2 neurons (160)
#     (213,),         # Single layer, 2d/3 neurons (213) - paper recommendation  
#     (100,),         # Single layer, 100 neurons (baseline)
#     (50, 50),       # Two layers, 50 neurons each
#     (107, 53),      # Two layers, d/3, d/6 pattern (~107, ~53)
#     (160, 80),      # Two layers, d/2, d/4 pattern (160, 80)
# ]

# results = []
# print("Testing different architectures...")

# for i, hidden_dims in enumerate(architectures):
#     print(f"Testing architecture {i+1}/{len(architectures)}: {hidden_dims}")
    
#     start_time = time.time()
#     model_arch = LassoNetClassifierCV(
#         hidden_dims=hidden_dims,
#         path_multiplier=1.02,
#         cv=3,  # Reduce CV folds for faster computation
#         random_state=42
#     )
    
#     model_arch.fit(X_train_scaled, y_train)
#     fitting_time_arch = time.time() - start_time
    
#     accuracy_arch = model_arch.score(X_test_scaled, y_test)
#     n_features = len(model_arch.best_selected_)
    
#     results.append({
#         'architecture': str(hidden_dims),
#         'accuracy': accuracy_arch,
#         'n_features': n_features,
#         'feature_pct': n_features/X.shape[1]*100,
#         'lambda': model_arch.best_lambda_,
#         'fitting_time': fitting_time_arch
#     })
    
#     print(f"  Accuracy: {accuracy_arch:.4f}, Features: {n_features} ({n_features/X.shape[1]*100:.1f}%), "
#           f"Lambda: {model_arch.best_lambda_:.6f}, Time: {fitting_time_arch:.1f}s")

# # Display results summary
# print(f"\n{'='*80}")
# print("ARCHITECTURE COMPARISON SUMMARY")
# print(f"{'='*80}")
# results_df = pd.DataFrame(results)
# print(results_df.round(4))

# # Find best architecture
# best_arch_idx = results_df['accuracy'].idxmax()
# best_arch = results_df.iloc[best_arch_idx]
# print(f"\nBest Architecture: {best_arch['architecture']}")
# print(f"Best Accuracy: {best_arch['accuracy']:.4f}")
# print(f"Features Selected: {best_arch['n_features']} ({best_arch['feature_pct']:.1f}%)")

# # ========================================
# # 9. Final Summary
# # ========================================

# print("\n" + "="*80)
# print("FINAL SUMMARY FOR ABC DATASET")
# print("="*80)

# print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features (20 descriptors × 16 scales)")
# print(f"Classes: {len(np.unique(y))} classes")

# print(f"\nBest Method: {best_method}")
# print(f"✓ Optimal alpha (lambda): {model_cv.best_lambda_ if best_method == 'CV' else best_lambda:.6f}")
# print(f"✓ Selected features: {len(best_selected)} out of {X.shape[1]} ({len(best_selected)/X.shape[1]*100:.1f}%)")
# print(f"✓ Test accuracy: {max(accuracy_cv, best_score):.4f}")
# print(f"✓ Dense model accuracy: {accuracy_dense:.4f}")

# print(f"\nTop 5 Most Important Descriptors:")
# if best_method == "CV":
#     top_descriptors = np.argsort(descriptor_weights_cv)[::-1][:5]
#     for i, desc_idx in enumerate(top_descriptors):
#         if descriptor_weights_cv[desc_idx] > 0:
#             weight = descriptor_weights_cv[desc_idx]
#             scales = int(descriptor_counts_cv[desc_idx])
#             print(f"{i+1}. {descriptor_names[desc_idx]:20s} | Weight: {weight:8.4f} | Scales: {scales:2d}/16")

# print(f"\nRecommendations:")
# print(f"• Selected {len(best_selected)} features achieve {max(accuracy_cv, best_score):.1%} accuracy")
# print(f"• Dense model shows {accuracy_dense - max(accuracy_cv, best_score):+.4f} improvement")
# print(f"• Consider using {best_arch['architecture']} architecture for optimal performance")
# print(f"• Feature selection reduces dimensionality by {100-len(best_selected)/X.shape[1]*100:.1f}%")

import numpy as np
import pandas as pd
import glob
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Install LassoNet if not already installed
# pip install lassonet

from lassonet import LassoNetClassifierCV, LassoNetClassifier, plot_path, plot_cv

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

# ABC Dataset structure helper functions
def get_descriptor_info(feature_idx, features_per_scale=20, n_scales=16):
    """Get descriptor type and scale for a given feature index"""
    # For ABC dataset: 320 features = 20 descriptors × 16 scales
    # Features are organized as: [desc0_scale0, desc0_scale1, ..., desc0_scale15, desc1_scale0, ...]
    descriptor_idx = feature_idx // n_scales
    scale_idx = feature_idx % n_scales
    
    # Ensure indices are within bounds
    descriptor_idx = min(descriptor_idx, features_per_scale - 1)
    scale_idx = min(scale_idx, n_scales - 1)
    
    return descriptor_idx, scale_idx

def analyze_selected_descriptors(selected_features, features_per_scale=20, n_scales=16, feature_importances=None):
    """Analyze which descriptors and scales were selected"""
    # Descriptor types for ABC dataset (20 descriptors × 16 scales = 320 features)
    descriptor_names = [
        'd1', 'd2', 'd3', 'd4',
        'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13',
        'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20'
    ]
    
    # Initialize tracking arrays
    descriptor_weights = np.zeros(features_per_scale)
    descriptor_counts = np.zeros(features_per_scale)
    scale_matrix = np.zeros((features_per_scale, n_scales))
    
    # Convert feature_importances to numpy array if it's a tensor
    if feature_importances is not None:
        if hasattr(feature_importances, 'cpu'):  # PyTorch tensor
            feature_importances = feature_importances.cpu().numpy()
        elif hasattr(feature_importances, 'numpy'):  # TensorFlow tensor
            feature_importances = feature_importances.numpy()
        else:
            feature_importances = np.array(feature_importances)
    
    for feat_idx in selected_features:
        descriptor_idx, scale_idx = get_descriptor_info(feat_idx, features_per_scale, n_scales)
        
        # Count selections
        descriptor_counts[descriptor_idx] += 1
        scale_matrix[descriptor_idx, scale_idx] = 1
        
        # Add importance weights if available
        if feature_importances is not None:
            descriptor_weights[descriptor_idx] += float(feature_importances[feat_idx])
        else:
            descriptor_weights[descriptor_idx] += 1
    
    return descriptor_names, descriptor_weights, descriptor_counts, scale_matrix

# Load actual ABC data
ssm_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/SSM_Challenge-ABC"
lb_path = "/home/cle/Work/ABC-Challenge/Dataset/Train/lb"

# Start with a subset for testing, then increase
print("Loading ABC Dataset...")
X, y = load_abc_data(ssm_path, lb_path, max_files=19)  # Start with fraction=0.1
print(f"Dataset shape: {X.shape}")
print(f"Classes distribution: {np.bincount(y)}")

# ========================================
# 2. Data Preprocessing
# ========================================

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# IMPORTANT: Normalize the data (LassoNet requires this)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# ========================================
# 3. Method 1: Cross-Validation Interface - devide training into groups/fold, each group folds try different alpha. Receive alpha for the best accuracy on test set
# Problem, LassoNet use hidden layer to check the final classified accuracy. 
# For example, increaseing alpha == zero out some features == higher accuarcy, this motivated the cross-validation to choose this alpha.
# But incase of ABC, (with a simple hidden layer) fully selected features == max accuracy ---> zero out some features == always smaller accuarcy. Remember that the goal is not accuracy, it is the speed!
# In case of ABC, it always have max performance on 20x16. The problem is that we choose less to optimize performance. 
# LassoNetCV really don't have motivation to select less (if the motivation is accuracy).
# Question 1? Architecture for ABC to reach higher accuracy with less featutred/descriptor seletecd?
# Question 2? Can we modify LassoNet so it take into account the speed too? Neural network with time-loss (performance) not accuracy(label - predict)? This doesn't make anysense, less featreus ~ better performance, beside we don't have such thing called "label-training time"!
# ========================================

# Add diagnostic information before training
print("\n" + "="*60)
print("DIAGNOSTIC INFORMATION")
print("="*60)
print(f"Data shape: {X.shape}")
print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
print(f"Data mean: {X.mean():.3f}, std: {X.std():.3f}")
print(f"After scaling - mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Class percentages: {np.bincount(y) / len(y) * 100}")

# Check if dataset is too easy (perfect separability)
from sklearn.linear_model import LogisticRegression
simple_model = LogisticRegression(max_iter=1000)
simple_model.fit(X_train_scaled, y_train)
simple_accuracy = simple_model.score(X_test_scaled, y_test)
print(f"Simple Logistic Regression accuracy: {simple_accuracy:.4f}")

if simple_accuracy > 0.98:
    print("⚠️  WARNING: Dataset appears to be linearly separable!")
    print("   This may explain why feature selection doesn't occur.")
    print("   Consider using a more challenging subset or different classes.")

print("\n" + "="*60)
print("Method 1: Cross-Validation Approach")
print("="*60)

# Track fitting time
start_time = time.time()

# Use LassoNetClassifierCV for automatic hyperparameter selection
model_cv = LassoNetClassifierCV(
    hidden_dims=(100, 200, 320, 400),      # 2d/3 for d=320 as per paper recommendation, this mean layer: 320x1 -> 213x1 -> 320x1 -> num_classesx1
    path_multiplier=1.05,    # Controls regularization path granularity
    M=10,                    # Hierarchy parameter (default works well)
    cv=5,                    # 5-fold cross-validation, devide the training set into 5 and train, result = average of those 5. Each have auto-assigned lambda-start. The more the better the slower.
    lambda_start="auto",     # auto=0
    random_state=42
)

# Fit the model (this will automatically find the best lambda)
print("Training LassoNet with cross-validation...")
path_cv = model_cv.fit(X_train_scaled, y_train)

fitting_time_cv = time.time() - start_time

# Evaluate the model
y_pred_cv = model_cv.predict(X_test_scaled)
accuracy_cv = accuracy_score(y_test, y_pred_cv)

# Results Summary for Method 1
print(f"\n{'='*40}")
print("METHOD 1 RESULTS (Cross-Validation)")
print(f"{'='*40}")
print(f"Fitting time: {fitting_time_cv:.2f} seconds")
print(f"Optimal alpha (lambda): {model_cv.best_lambda_:.6f}")
print(f"Selected features: {len(model_cv.best_selected_)} out of {X.shape[1]}")
print(f"Feature selection percentage: {len(model_cv.best_selected_)/X.shape[1]*100:.1f}%")
print(f"Best CV score: {model_cv.best_cv_score_:.4f}")
print(f"Test accuracy: {accuracy_cv:.4f}")

# Check if feature selection actually occurred
if len(model_cv.best_selected_) == X.shape[1]:
    print("⚠️  WARNING: No feature selection occurred! All features were selected.")
    print("   This suggests lambda is too small. Consider:")
    print("   - Using a larger lambda_start")
    print("   - Increasing path_multiplier")
    print("   - Check if data scaling is appropriate")

# Analyze selected descriptors for Method 1
try:
    descriptor_names, descriptor_weights_cv, descriptor_counts_cv, scale_matrix_cv = analyze_selected_descriptors(
        model_cv.best_selected_, feature_importances=model_cv.feature_importances_
    )
    
    print(f"\nDescriptor Analysis (Total Importance/Weight):")
    print("-" * 60)
    sorted_indices = np.argsort(descriptor_weights_cv)[::-1]
    for i, desc_idx in enumerate(sorted_indices):
        if descriptor_weights_cv[desc_idx] > 0:
            n_scales = int(descriptor_counts_cv[desc_idx])
            weight = descriptor_weights_cv[desc_idx]
            print(f"{i+1:2d}. {descriptor_names[desc_idx]:20s} | Weight: {weight:8.4f} | Scales: {n_scales:2d}/16")
            
            # Show all 20 descriptors
            if i >= 19:
                break
                
except Exception as e:
    print(f"Error in descriptor analysis: {e}")
    print("Continuing with basic analysis...")
    descriptor_weights_cv = np.ones(20)  # Fallback
    descriptor_counts_cv = np.ones(20)
    scale_matrix_cv = np.ones((20, 16))

# ========================================
# 4. Method 2: Manual Path Exploration - choose alpha according to path
# ========================================

print("\n" + "="*60)
print("Method 2: Manual Path Exploration")
print("="*60)


start_time = time.time()

# Model
model_base = LassoNetClassifier(
    hidden_dims=(100, 200, 320, 400),
    M=10,
    random_state=42
)

print("Computing regularization path...")
#Create path with default lambda_start and path_muttiplier (default when we generate the LassoNetClassifier)
#Path λ (regularization strength) = [lamda_start=0.0(default), 0.01, 0.05,.... 5.0]. With each λ thenetwork reducate/unconnected a set of features until it uncoonected all!
path = model_base.path(X_train_scaled, y_train, return_state_dicts=True) 

fitting_time_base = time.time() - start_time

# Find best model from path based on validation score
best_idx = 0
best_score = 0
for i, checkpoint in enumerate(path):
    model_base.load(checkpoint.state_dict)
    score = model_base.score(X_test_scaled, y_test)
    if score > best_score:
        best_score = score
        best_idx = i

# Load best model
model_base.load(path[best_idx].state_dict)
best_lambda = path[best_idx].lambda_
selected_features_base = path[best_idx].selected

# Results Summary for Method 2
print(f"\n{'='*40}")
print("METHOD 2 RESULTS (Manual Path)")
print(f"{'='*40}")
print(f"Fitting time: {fitting_time_base:.2f} seconds")
print(f"Optimal alpha (lambda): {best_lambda:.6f}")
print(f"Selected features: {len(selected_features_base)} out of {X.shape[1]}")
print(f"Feature selection percentage: {len(selected_features_base)/X.shape[1]*100:.1f}%")
print(f"Test accuracy: {best_score:.4f}")

# Analyze selected descriptors for Method 2
try:
    descriptor_names, descriptor_weights_base, descriptor_counts_base, scale_matrix_base = analyze_selected_descriptors(
        selected_features_base
    )

    print(f"\nDescriptor Analysis (Selection Count):")
    print("-" * 60)
    sorted_indices = np.argsort(descriptor_counts_base)[::-1]
    for i, desc_idx in enumerate(sorted_indices):
        if descriptor_counts_base[desc_idx] > 0:
            n_scales = int(descriptor_counts_base[desc_idx])
            print(f"{i+1:2d}. {descriptor_names[desc_idx]:20s} | Count: {n_scales:2d}/16 scales")
            
            # Show all 20 descriptors
            if i >= 19:
                break
except Exception as e:
    print(f"Error in descriptor analysis: {e}")
    print("Continuing with basic analysis...")
    descriptor_weights_base = np.ones(20)  # Fallback
    descriptor_counts_base = np.ones(20)
    scale_matrix_base = np.ones((20, 16))

# ========================================
# 5. Visualization
# ========================================

print("\n" + "="*60)
print("Visualization")
print("="*60)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Cross-validation plot (with error handling)
plt.subplot(2, 4, 1)
try:
    if hasattr(model_cv, 'cv_scores_') and model_cv.cv_scores_ is not None:
        plot_cv(model_cv, X_test_scaled, y_test)
        plt.title("CV Results (Method 1)")
    else:
        plt.text(0.5, 0.5, 'CV plot not available\n(no feature selection)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("CV Results (Method 1) - N/A")
except Exception as e:
    plt.text(0.5, 0.5, f'CV plot error:\n{str(e)[:50]}...', 
            ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("CV Results (Method 1) - Error")

# 2. Regularization path plot (with error handling)
plt.subplot(2, 4, 2)
try:
    if len(path) > 1 and any(len(checkpoint.selected) < X.shape[1] for checkpoint in path):
        plot_path(model_base, X_test_scaled, y_test)
        plt.title("Regularization Path (Method 2)")
    else:
        plt.text(0.5, 0.5, 'Path plot not available\n(no feature selection)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Regularization Path (Method 2) - N/A")
except Exception as e:
    plt.text(0.5, 0.5, f'Path plot error:\n{str(e)[:50]}...', 
            ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Regularization Path (Method 2) - Error")

# 3. Heatmap of selected descriptors/scales (Method 1)
plt.subplot(2, 4, 3)
sns.heatmap(scale_matrix_cv, 
            xticklabels=range(1, 17), 
            yticklabels=[name[:10] for name in descriptor_names],
            cmap='Reds', 
            cbar_kws={'label': 'Selected'})
plt.title("Selected Features Heatmap\n(Method 1: CV)")
plt.xlabel("Scale Index")
plt.ylabel("Descriptor")

# 4. Column graph of descriptor weights (Method 1)
plt.subplot(2, 4, 4)
sorted_indices = np.argsort(descriptor_weights_cv)[::-1]
top_10_indices = sorted_indices[:10]
top_10_weights = descriptor_weights_cv[top_10_indices]
top_10_names = [descriptor_names[i][:10] for i in top_10_indices]

plt.bar(range(len(top_10_weights)), top_10_weights)
plt.xticks(range(len(top_10_names)), top_10_names, rotation=45)
plt.title("Top 10 Descriptor Weights\n(Method 1: CV)")
plt.ylabel("Total Weight")

# 5. Heatmap of selected descriptors/scales (Method 2)
plt.subplot(2, 4, 7)
sns.heatmap(scale_matrix_base, 
            xticklabels=range(1, 17), 
            yticklabels=[name[:10] for name in descriptor_names],
            cmap='Blues', 
            cbar_kws={'label': 'Selected'})
plt.title("Selected Features Heatmap\n(Method 2: Manual)")
plt.xlabel("Scale Index")
plt.ylabel("Descriptor")

# 6. Column graph of descriptor counts (Method 2)
plt.subplot(2, 4, 8)
sorted_indices = np.argsort(descriptor_counts_base)[::-1]
top_10_indices = sorted_indices[:10]
top_10_counts = descriptor_counts_base[top_10_indices]
top_10_names = [descriptor_names[i][:10] for i in top_10_indices]

plt.bar(range(len(top_10_counts)), top_10_counts)
plt.xticks(range(len(top_10_names)), top_10_names, rotation=45)
plt.title("Top 10 Descriptor Counts\n(Method 2: Manual)")
plt.ylabel("Number of Scales Selected")

# 7. Feature importance comparison
plt.subplot(2, 4, 5)
try:
    if hasattr(model_cv, 'feature_importances_') and model_cv.feature_importances_ is not None:
        importances = model_cv.feature_importances_
        if hasattr(importances, 'cpu'):  # Handle tensor
            importances = importances.cpu().numpy()
        plt.plot(sorted(importances), 'o-', alpha=0.7)
        plt.title("Feature Importance Distribution\n(Method 1)")
        plt.xlabel("Feature Rank")
        plt.ylabel("Importance")
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Feature Importance - N/A")
except Exception as e:
    plt.text(0.5, 0.5, f'Importance error:\n{str(e)[:30]}...', 
            ha='center', va='center', transform=plt.gca().transAxes)
    plt.title("Feature Importance - Error")

# 8. Model comparison
plt.subplot(2, 4, 6)
methods = ['CV', 'Manual']
accuracies = [accuracy_cv, best_score]
times = [fitting_time_cv, fitting_time_base]
selected_counts = [len(model_cv.best_selected_), len(selected_features_base)]

x = np.arange(len(methods))
width = 0.25

plt.bar(x - width, accuracies, width, label='Accuracy', alpha=0.7)
plt.bar(x, np.array(times)/max(times), width, label='Relative Time', alpha=0.7)
plt.bar(x + width, np.array(selected_counts)/X.shape[1], width, label='Feature %', alpha=0.7)

plt.xlabel('Method')
plt.ylabel('Normalized Score')
plt.title('Method Comparison')
plt.xticks(x, methods)
plt.legend()

plt.tight_layout()
plt.savefig('lassonet_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ========================================
# 7. Train Dense Model on Selected Features
# ========================================

print("\n" + "="*60)
print("Dense Model on Selected Features")
print("="*60)

# Use the better performing method
best_model = model_cv if accuracy_cv >= best_score else model_base
best_selected = model_cv.best_selected_ if accuracy_cv >= best_score else selected_features_base
best_method = "CV" if accuracy_cv >= best_score else "Manual"

print(f"Using {best_method} method results for dense training")

# Train a dense neural network on only the selected features
X_train_selected = X_train_scaled[:, best_selected]
X_test_selected = X_test_scaled[:, best_selected]

# Train dense model
start_time = time.time()
model_dense = LassoNetClassifier(hidden_dims=(100,), M=10)
model_dense.fit(X_train_selected, y_train, dense_only=True)
dense_fitting_time = time.time() - start_time

# Evaluate dense model
y_pred_dense = model_dense.predict(X_test_selected)
accuracy_dense = accuracy_score(y_test, y_pred_dense)

print(f"Dense model training time: {dense_fitting_time:.2f} seconds")
print(f"Dense model accuracy on {len(best_selected)} selected features: {accuracy_dense:.4f}")
print(f"Improvement over LassoNet: {accuracy_dense - max(accuracy_cv, best_score):.4f}")

# ========================================
# 8. Compare Different Hidden Layer Sizes
# ========================================

print("\n" + "="*60)
print("Comparing Different Architectures")
print("="*60)

# Test different architectures as suggested in the paper
d = X.shape[1]  # Number of features = 320
architectures = [
    (d//3,),        # Single layer, d/3 neurons (~107)
    (d//2,),        # Single layer, d/2 neurons (160)
    (213,),         # Single layer, 2d/3 neurons (213) - paper recommendation  
    (100,),         # Single layer, 100 neurons (baseline)
    (50, 50),       # Two layers, 50 neurons each
    (107, 53),      # Two layers, d/3, d/6 pattern (~107, ~53)
    (160, 80),      # Two layers, d/2, d/4 pattern (160, 80)
]

results = []
print("Testing different architectures...")

for i, hidden_dims in enumerate(architectures):
    print(f"Testing architecture {i+1}/{len(architectures)}: {hidden_dims}")
    
    start_time = time.time()
    model_arch = LassoNetClassifierCV(
        hidden_dims=hidden_dims,
        path_multiplier=1.02,
        cv=3,  # Reduce CV folds for faster computation
        random_state=42
    )
    
    model_arch.fit(X_train_scaled, y_train)
    fitting_time_arch = time.time() - start_time
    
    accuracy_arch = model_arch.score(X_test_scaled, y_test)
    n_features = len(model_arch.best_selected_)
    
    results.append({
        'architecture': str(hidden_dims),
        'accuracy': accuracy_arch,
        'n_features': n_features,
        'feature_pct': n_features/X.shape[1]*100,
        'lambda': model_arch.best_lambda_,
        'fitting_time': fitting_time_arch
    })
    
    print(f"  Accuracy: {accuracy_arch:.4f}, Features: {n_features} ({n_features/X.shape[1]*100:.1f}%), "
          f"Lambda: {model_arch.best_lambda_:.6f}, Time: {fitting_time_arch:.1f}s")

# Display results summary
print(f"\n{'='*80}")
print("ARCHITECTURE COMPARISON SUMMARY")
print(f"{'='*80}")
results_df = pd.DataFrame(results)
print(results_df.round(4))

# Find best architecture
best_arch_idx = results_df['accuracy'].idxmax()
best_arch = results_df.iloc[best_arch_idx]
print(f"\nBest Architecture: {best_arch['architecture']}")
print(f"Best Accuracy: {best_arch['accuracy']:.4f}")
print(f"Features Selected: {best_arch['n_features']} ({best_arch['feature_pct']:.1f}%)")

# ========================================
# 9. Final Summary
# ========================================

print("\n" + "="*80)
print("FINAL SUMMARY FOR ABC DATASET")
print("="*80)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features (20 descriptors × 16 scales)")
print(f"Classes: {len(np.unique(y))} classes")

print(f"\nBest Method: {best_method}")
print(f"✓ Optimal alpha (lambda): {model_cv.best_lambda_ if best_method == 'CV' else best_lambda:.6f}")
print(f"✓ Selected features: {len(best_selected)} out of {X.shape[1]} ({len(best_selected)/X.shape[1]*100:.1f}%)")
print(f"✓ Test accuracy: {max(accuracy_cv, best_score):.4f}")
print(f"✓ Dense model accuracy: {accuracy_dense:.4f}")

print(f"\nTop 20 Most Important Descriptors:")
if best_method == "CV":
    top_descriptors = np.argsort(descriptor_weights_cv)[::-1][:20]
    for i, desc_idx in enumerate(top_descriptors):
        if descriptor_weights_cv[desc_idx] > 0:
            weight = descriptor_weights_cv[desc_idx]
            scales = int(descriptor_counts_cv[desc_idx])
            print(f"{i+1:2d}. {descriptor_names[desc_idx]:20s} | Weight: {weight:8.4f} | Scales: {scales:2d}/16")

print(f"\nRecommendations:")
print(f"• Selected {len(best_selected)} features achieve {max(accuracy_cv, best_score):.1%} accuracy")
print(f"• Dense model shows {accuracy_dense - max(accuracy_cv, best_score):+.4f} improvement")
print(f"• Consider using {best_arch['architecture']} architecture for optimal performance")
print(f"• Feature selection reduces dimensionality by {100-len(best_selected)/X.shape[1]*100:.1f}%")
