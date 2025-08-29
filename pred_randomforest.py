import os
import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import time
from utils import DataGenerator

# Create results folder
RESULTS_FOLDER = "result_randomforest"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
print(f"Results will be saved to: {RESULTS_FOLDER}")

class RandomForestEdgeClassifier:
    """
    Random Forest classifier for edge/non-edge classification using ABC dataset
    """
    
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=None, 
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 random_state=42,
                 n_jobs=-1):
        """
        Initialize Random Forest classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            random_state: Random state for reproducibility
            n_jobs: Number of jobs to run in parallel
        """
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.is_fitted = False
        self.feature_names = None
        self.training_time = None
        
    def prepare_data_from_generator(self, data_generator, max_batches=None):
        """
        Extract data from DataGenerator and prepare for sklearn
        
        Args:
            data_generator: DataGenerator instance
            max_batches: Maximum number of batches to process (None for all)
            
        Returns:
            X, y: Features and labels as numpy arrays
        """
        print("Extracting data from generator...")
        
        X_list = []
        y_list = []
        
        num_batches = len(data_generator)
        if max_batches:
            num_batches = min(num_batches, max_batches)
        
        print(f"Processing {num_batches} batches...")
        
        # Try to get data directly from the generator's loaded data first
        if hasattr(data_generator, 'all_data') and data_generator.all_data is not None:
            print("Using pre-loaded data from generator...")
            
            # Limit the amount of data if max_batches is specified
            if max_batches:
                total_samples = min(len(data_generator.all_data), max_batches * data_generator.batch_size)
                indices = np.random.choice(len(data_generator.all_data), total_samples, replace=False)
                X = data_generator.all_data[indices]
                y = data_generator.all_labels[indices]
            else:
                X = data_generator.all_data
                y = data_generator.all_labels
            
            # Handle different output formats
            if data_generator.output_format == 'dict':
                # For dictionary format, we need to flatten
                X_flat = X.reshape(X.shape[0], -1)
            else:
                # Already flattened format
                X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
            
            print(f"Data shape: X={X_flat.shape}, y={y.shape}")
            print(f"Class distribution: {np.unique(y, return_counts=True)}")
            
            return X_flat, y
        
        # Fallback to batch-by-batch loading
        successful_batches = 0
        for i in tqdm(range(num_batches), desc="Loading batches"):
            try:
                batch_X, batch_y = data_generator[i]
                
                # Check if batch is empty
                if batch_X is None or batch_y is None:
                    print(f"Batch {i}: Empty batch, skipping...")
                    continue
                    
                # Handle different output formats
                if isinstance(batch_X, dict):
                    # Dictionary format - concatenate all scales
                    batch_features = []
                    for scale_key in sorted(batch_X.keys()):
                        if batch_X[scale_key].size > 0:  # Check if not empty
                            batch_features.append(batch_X[scale_key])
                    
                    if batch_features:
                        batch_X_flat = np.concatenate(batch_features, axis=1)
                    else:
                        print(f"Batch {i}: No valid features, skipping...")
                        continue
                else:
                    # Already flattened format
                    if batch_X.size == 0:
                        print(f"Batch {i}: Empty features, skipping...")
                        continue
                    batch_X_flat = batch_X
                
                # Check dimensions
                if len(batch_X_flat.shape) == 1:
                    batch_X_flat = batch_X_flat.reshape(1, -1)
                if len(batch_y.shape) == 2 and batch_y.shape[1] == 1:
                    batch_y = batch_y.flatten()
                
                # Check if batch has valid data
                if batch_X_flat.shape[0] > 0 and batch_y.shape[0] > 0:
                    X_list.append(batch_X_flat)
                    y_list.append(batch_y)
                    successful_batches += 1
                    print(f"Batch {i}: Successfully loaded {batch_X_flat.shape[0]} samples")
                else:
                    print(f"Batch {i}: Invalid dimensions, skipping...")
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Successfully processed {successful_batches}/{num_batches} batches")
        
        if not X_list:
            raise ValueError("No data was successfully loaded from any batch")
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        print(f"Final data shape: X={X.shape}, y={y.shape}")
        print(f"Class distribution: {np.unique(y, return_counts=True)}")
        
        return X, y
    
    def train(self, train_generator, max_batches=None):
        """
        Train the Random Forest model
        
        Args:
            train_generator: DataGenerator for training data
            max_batches: Maximum batches to use for training
        """
        print("=== Training Random Forest Classifier ===")
        
        # Prepare data
        X_train, y_train = self.prepare_data_from_generator(train_generator, max_batches)
        
        # Store generator info for better feature naming
        self.target_scales = getattr(train_generator, 'target_scales', 16)
        self.target_features = getattr(train_generator, 'target_features', 20)
        
        # Create feature names
        self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train model
        print("Training Random Forest...")
        start_time = time.time()
        
        self.rf.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        # Training accuracy
        train_pred = self.rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        return self
    
    def evaluate(self, test_generator, max_batches=None):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: DataGenerator for test data
            max_batches: Maximum batches to use for evaluation
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        print("=== Evaluating Random Forest Classifier ===")
        
        # Prepare test data
        X_test, y_test = self.prepare_data_from_generator(test_generator, max_batches)
        
        # Predictions
        print("Making predictions...")
        start_time = time.time()
        y_pred = self.rf.predict(X_test)
        y_pred_proba = self.rf.predict_proba(X_test)
        prediction_time = time.time() - start_time
        
        print(f"Prediction completed in {prediction_time:.2f} seconds")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # IoU calculation
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        iou = tp / (tp + fp + fn)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'iou': iou,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'prediction_time': prediction_time,
            'n_test_samples': len(y_test)
        }
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"MCC:       {mcc:.4f}")
        print(f"IoU:       {iou:.4f}")
        print(f"Test samples: {len(y_test)}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return metrics
    
    def plot_feature_importance(self, top_n=20, filename="feature_importance.png", 
                               use_permutation=False, test_data=None):
        """
        Plot feature importance with meaningful SSM descriptor names
        
        Args:
            top_n: Number of top features to show
            filename: Filename for saving the plot
            use_permutation: If True, use permutation importance (requires test_data)
            test_data: Tuple of (X_test, y_test) for permutation importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before plotting feature importance")
        
        # Get feature importance
        if use_permutation and test_data is not None:
            print("Calculating permutation importance (this may take a while)...")
            X_test, y_test = test_data
            perm_result = permutation_importance(self.rf, X_test, y_test, 
                                               n_repeats=5, random_state=42)
            importance = perm_result.importances_mean
            importance_std = perm_result.importances_std
            method_name = "Permutation"
        else:
            importance = self.rf.feature_importances_
            importance_std = None
            method_name = "Gini"
        
        # Create meaningful feature names for SSM descriptors
        def get_ssm_feature_name(feature_idx):
            """Convert flattened feature index to SSM descriptor name"""
            scale = feature_idx // 20  # Which of the 16 scales
            descriptor = (feature_idx % 20) + 1  # Which of the 20 descriptors (1-indexed)
            return f"Scale{scale}_Desc{descriptor}"
        
        # Generate feature names
        if hasattr(self, 'target_scales') and hasattr(self, 'target_features'):
            feature_names = [get_ssm_feature_name(i) for i in range(len(importance))]
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Sort by importance (descending order)
        indices = np.argsort(importance)[::-1][:top_n]
        
        # Create plot
        plt.figure(figsize=(14, 8))
        plt.title(f'Top {top_n} Feature Importances - Random Forest ({method_name} Method)\n(SSM Descriptors across Multiple Scales)')
        
        # Create bar plot with error bars if available
        bars = plt.bar(range(top_n), importance[indices], color='steelblue', alpha=0.7,
                      yerr=importance_std[indices] if importance_std is not None else None,
                      capsize=3)
        
        # Customize x-axis labels
        feature_labels = [feature_names[i] for i in indices]
        plt.xticks(range(top_n), feature_labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, idx) in enumerate(zip(bars, indices)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance[idx]:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('SSM Features (Scale_Descriptor)')
        plt.ylabel(f'Feature Importance ({method_name})')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save to results folder
        save_path = os.path.join(RESULTS_FOLDER, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        # Print detailed analysis
        print(f"\n=== How {method_name} Importance is Calculated ===")
        if use_permutation:
            print("Permutation Importance:")
            print("- Measures how much model performance drops when feature is shuffled")
            print("- More robust than Gini importance")
            print("- Higher values = feature is more important for predictions")
        else:
            print("Gini Importance (Mean Decrease Impurity):")
            print("- Measures how much each feature decreases node impurity")
            print("- Based on how often feature is used and how much it improves splits")
            print("- Higher values = feature creates purer splits more frequently")
        
        # Print top features analysis
        print(f"\n=== Top {min(10, top_n)} Most Important Features ===")
        for i, idx in enumerate(indices[:min(10, top_n)]):
            scale = idx // 20
            descriptor = (idx % 20) + 1
            std_str = f" ± {importance_std[idx]:.4f}" if importance_std is not None else ""
            print(f"{i+1:2d}. {feature_names[idx]:15s} - Importance: {importance[idx]:.4f}{std_str} (Scale {scale}, Descriptor {descriptor})")
        
        # Analyze by scale
        scale_importance = {}
        for idx in indices:
            scale = idx // 20
            if scale not in scale_importance:
                scale_importance[scale] = 0
            scale_importance[scale] += importance[idx]
        
        print(f"\n=== Importance by Scale (Top {top_n} features) ===")
        sorted_scales = sorted(scale_importance.items(), key=lambda x: x[1], reverse=True)
        for scale, total_imp in sorted_scales:
            print(f"Scale {scale:2d}: {total_imp:.4f}")
        
        # Analyze by descriptor
        desc_importance = {}
        for idx in indices:
            descriptor = (idx % 20) + 1
            if descriptor not in desc_importance:
                desc_importance[descriptor] = 0
            desc_importance[descriptor] += importance[idx]
        
        print(f"\n=== Importance by Descriptor (Top {top_n} features) ===")
        sorted_descriptors = sorted(desc_importance.items(), key=lambda x: x[1], reverse=True)
        for desc, total_imp in sorted_descriptors[:10]:  # Show top 10 descriptors
            print(f"Descriptor {desc:2d}: {total_imp:.4f}")
        
    def plot_feature_importance_by_scale(self, filename="feature_importance_by_scale.png"):
        """
        Plot feature importance grouped by scale, showing all 20 descriptors per scale
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before plotting feature importance")
        
        importance = self.rf.feature_importances_
        
        # Organize importance by scale and descriptor
        scale_data = {}
        for i, imp in enumerate(importance):
            scale = i // 20  # Which scale (0-15)
            descriptor = i % 20  # Which descriptor (0-19)
            
            if scale not in scale_data:
                scale_data[scale] = np.zeros(20)
            scale_data[scale][descriptor] = imp
        
        # Create the grouped bar plot
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Set up the bar positions
        num_scales = len(scale_data)
        bar_width = 0.04  # Width of each descriptor bar
        descriptor_positions = np.arange(20)  # Position for each descriptor (0-19)
        
        # Colors for different scales
        colors = plt.cm.tab20(np.linspace(0, 1, num_scales))
        
        # Plot bars for each scale
        for scale_idx, (scale, desc_importances) in enumerate(sorted(scale_data.items())):
            # Offset position for this scale
            positions = descriptor_positions + (scale_idx - num_scales/2 + 0.5) * bar_width
            
            bars = ax.bar(positions, desc_importances, bar_width, 
                         label=f'Scale {scale}', color=colors[scale_idx], alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('SSM Descriptors (1-20)')
        ax.set_ylabel('Feature Importance')
        ax.set_title('Feature Importance by Scale - All 20 SSM Descriptors\n(Each color represents a different scale)')
        ax.set_xticks(descriptor_positions)
        ax.set_xticklabels([f'Desc{i+1}' for i in range(20)])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(RESULTS_FOLDER, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance by scale plot saved to {save_path}")
        plt.show()
        
        # Print summary statistics
        print("\n=== Feature Importance Summary by Scale ===")
        total_importance_by_scale = {}
        for scale, desc_importances in sorted(scale_data.items()):
            total_imp = np.sum(desc_importances)
            max_desc = np.argmax(desc_importances) + 1
            max_imp = np.max(desc_importances)
            total_importance_by_scale[scale] = total_imp
            
            print(f"Scale {scale:2d}: Total = {total_imp:.4f}, "
                  f"Best descriptor = Desc{max_desc:2d} ({max_imp:.4f})")
        
        print("\n=== Feature Importance Summary by Descriptor ===")
        descriptor_totals = np.zeros(20)
        for scale, desc_importances in scale_data.items():
            descriptor_totals += desc_importances
        
        # Sort descriptors by total importance
        desc_ranking = np.argsort(descriptor_totals)[::-1]
        for rank, desc_idx in enumerate(desc_ranking[:10]):  # Top 10
            total_imp = descriptor_totals[desc_idx]
            print(f"Descriptor {desc_idx+1:2d}: Total = {total_imp:.4f} (Rank {rank+1})")
        
        return scale_data, descriptor_totals

    def plot_heatmap_importance(self, filename="importance_heatmap.png"):
        """
        Plot feature importance as a heatmap (Scale x Descriptor)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before plotting feature importance")
        
        importance = self.rf.feature_importances_
        
        # Create 2D array: rows = scales, columns = descriptors
        importance_matrix = np.zeros((16, 20))  # 16 scales x 20 descriptors
        
        for i, imp in enumerate(importance):
            scale = i // 20
            descriptor = i % 20
            importance_matrix[scale, descriptor] = imp
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        
        # Use seaborn for better heatmap
        sns.heatmap(importance_matrix, 
                   xticklabels=[f'Desc{i+1}' for i in range(20)],
                   yticklabels=[f'Scale{i}' for i in range(16)],
                   cmap='YlOrRd', 
                   annot=False,  # Set to True if you want values in cells
                   fmt='.3f',
                   cbar_kws={'label': 'Feature Importance'})
        
        plt.title('Feature Importance Heatmap\n(Scales vs SSM Descriptors)')
        plt.xlabel('SSM Descriptors')
        plt.ylabel('Scales')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(RESULTS_FOLDER, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance heatmap saved to {save_path}")
        plt.show()
        
    def plot_descriptor_importance_summary(self, filename="descriptor_importance_summary.png"):
        """
        Plot total importance for each of the 20 descriptors (summed across all 16 scales)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before plotting feature importance")
        
        importance = self.rf.feature_importances_
        
        # Sum importance for each descriptor across all scales
        descriptor_totals = np.zeros(20)
        for i, imp in enumerate(importance):
            descriptor = i % 20  # Which descriptor (0-19)
            descriptor_totals[descriptor] += imp
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Create bar plot
        bars = plt.bar(range(20), descriptor_totals, color='steelblue', alpha=0.8, edgecolor='navy')
        
        # Customize the plot
        plt.title('SSM Descriptor Importance Summary\n(Total importance across all 16 scales)', fontsize=16)
        plt.xlabel('SSM Descriptors', fontsize=12)
        plt.ylabel('Total Feature Importance', fontsize=12)
        plt.xticks(range(20), [f'Desc{i+1}' for i in range(20)])
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, total_imp) in enumerate(zip(bars, descriptor_totals)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{total_imp:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(RESULTS_FOLDER, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Descriptor importance summary plot saved to {save_path}")
        plt.show()
        
        # Print ranking
        print("\n=== SSM Descriptor Importance Ranking ===")
        print("(Total importance summed across all 16 scales)")
        
        # Sort by total importance
        desc_ranking = np.argsort(descriptor_totals)[::-1]
        for rank, desc_idx in enumerate(desc_ranking):
            total_imp = descriptor_totals[desc_idx]
            percentage = (total_imp / np.sum(descriptor_totals)) * 100
            print(f"{rank+1:2d}. Descriptor {desc_idx+1:2d}: {total_imp:.4f} ({percentage:.1f}%)")
        
        return descriptor_totals, desc_ranking

    def export_importance_data(self, base_filename="ssm_importance_data"):
        """
        Export SSM descriptor importance data to multiple formats for later analysis
        
        Args:
            base_filename: Base name for output files (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before exporting importance data")
        
        importance = self.rf.feature_importances_
        
        # 1. Calculate descriptor totals (summed across scales)
        descriptor_totals = np.zeros(20)
        for i, imp in enumerate(importance):
            descriptor = i % 20
            descriptor_totals[descriptor] += imp
        
        # 2. Create detailed DataFrame with all scale-descriptor combinations
        detailed_data = []
        for i, imp in enumerate(importance):
            scale = i // 20
            descriptor = (i % 20) + 1  # 1-indexed for readability
            detailed_data.append({
                'Feature_Index': i,
                'Scale': scale,
                'Descriptor': descriptor,
                'Feature_Name': f'Scale{scale}_Desc{descriptor}',
                'Gini_Importance': imp
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # 3. Create summary DataFrame (descriptor totals)
        desc_ranking = np.argsort(descriptor_totals)[::-1]
        summary_data = []
        for rank, desc_idx in enumerate(desc_ranking):
            total_imp = descriptor_totals[desc_idx]
            percentage = (total_imp / np.sum(descriptor_totals)) * 100
            summary_data.append({
                'Rank': rank + 1,
                'Descriptor': desc_idx + 1,  # 1-indexed
                'Total_Importance': total_imp,
                'Percentage': percentage
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 4. Export to multiple formats
        
        # CSV files
        detailed_csv = os.path.join(RESULTS_FOLDER, f"{base_filename}_detailed.csv")
        summary_csv = os.path.join(RESULTS_FOLDER, f"{base_filename}_summary.csv")
        detailed_df.to_csv(detailed_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        
        # JSON files
        detailed_json = os.path.join(RESULTS_FOLDER, f"{base_filename}_detailed.json")
        summary_json = os.path.join(RESULTS_FOLDER, f"{base_filename}_summary.json")
        detailed_df.to_json(detailed_json, orient='records', indent=2)
        summary_df.to_json(summary_json, orient='records', indent=2)
        
        # Excel file with multiple sheets
        excel_file = os.path.join(RESULTS_FOLDER, f"{base_filename}.xlsx")
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Descriptor_Summary', index=False)
            detailed_df.to_excel(writer, sheet_name='All_Features', index=False)
            
            # Add scale-wise summary
            scale_summary = []
            for scale in range(16):
                scale_mask = detailed_df['Scale'] == scale
                scale_total = detailed_df[scale_mask]['Gini_Importance'].sum()
                scale_summary.append({
                    'Scale': scale,
                    'Total_Importance': scale_total,
                    'Percentage': (scale_total / importance.sum()) * 100
                })
            
            scale_df = pd.DataFrame(scale_summary)
            scale_df.to_excel(writer, sheet_name='Scale_Summary', index=False)
        
        # Simple text file for quick reference
        txt_file = os.path.join(RESULTS_FOLDER, f"{base_filename}_summary.txt")
        with open(txt_file, 'w') as f:
            f.write("SSM Descriptor Importance Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write("Descriptor Ranking (Total across all 16 scales):\n")
            f.write("-" * 50 + "\n")
            for rank, desc_idx in enumerate(desc_ranking):
                total_imp = descriptor_totals[desc_idx]
                percentage = (total_imp / np.sum(descriptor_totals)) * 100
                f.write(f"{rank+1:2d}. Descriptor {desc_idx+1:2d}: {total_imp:.6f} ({percentage:.2f}%)\n")
            
            f.write(f"\nTotal Importance Sum: {np.sum(descriptor_totals):.6f}\n")
            f.write(f"Number of Features: {len(importance)}\n")
            f.write(f"Number of Scales: 16\n")
            f.write(f"Number of Descriptors: 20\n")
        
        # Print export summary
        print(f"\n=== Importance Data Exported ===")
        print(f"Files saved to: {RESULTS_FOLDER}/")
        print(f"1. {base_filename}_summary.csv        - Descriptor ranking & totals")
        print(f"2. {base_filename}_detailed.csv       - All 320 features with importance")
        print(f"3. {base_filename}_summary.json       - JSON format descriptor summary")
        print(f"4. {base_filename}_detailed.json      - JSON format all features")
        print(f"5. {base_filename}.xlsx               - Excel file with multiple sheets")
        print(f"6. {base_filename}_summary.txt        - Simple text summary")
        
        return {
            'detailed_df': detailed_df,
            'summary_df': summary_df,
            'descriptor_totals': descriptor_totals,
            'ranking': desc_ranking
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, filename="confusion_matrix.png"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            filename: Filename for saving the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Edge', 'Edge'],
                    yticklabels=['Non-Edge', 'Edge'])
        plt.title('Confusion Matrix - Random Forest')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save to results folder
        save_path = os.path.join(RESULTS_FOLDER, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def hyperparameter_tuning(self, train_generator, max_batches=None, cv=3):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            train_generator: Training data generator
            max_batches: Maximum batches for training
            cv: Number of cross-validation folds
        """
        print("=== Hyperparameter Tuning ===")
        
        # Prepare data
        X_train, y_train = self.prepare_data_from_generator(train_generator, max_batches)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Grid search
        print("Performing grid search...")
        grid_search = GridSearchCV(
            self.rf, param_grid, 
            cv=cv, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.rf = grid_search.best_estimator_
        self.is_fitted = True
        
        return grid_search.best_params_
    
    def save_model(self, filename="random_forest_edge_classifier.pkl"):
        """Save the trained model to results folder"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        filepath = os.path.join(RESULTS_FOLDER, filename)
        joblib.dump(self.rf, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename="random_forest_edge_classifier.pkl"):
        """Load a trained model from results folder"""
        filepath = os.path.join(RESULTS_FOLDER, filename)
        self.rf = joblib.load(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
    
    def save_metrics_report(self, metrics, filename="evaluation_metrics.txt"):
        """
        Save evaluation metrics to a text file
        
        Args:
            metrics: Dictionary with evaluation metrics
            filename: Name of the file to save metrics
        """
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        with open(filepath, 'w') as f:
            f.write("=== Random Forest Edge Classification Results ===\n\n")
            f.write(f"Training Time: {getattr(self, 'training_time', 'N/A'):.2f} seconds\n")
            f.write(f"Prediction Time: {metrics.get('prediction_time', 'N/A'):.2f} seconds\n")
            f.write(f"Test Samples: {metrics.get('n_test_samples', 'N/A')}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-score:  {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"MCC:       {metrics.get('mcc', 0):.4f}\n")
            f.write(f"IoU:       {metrics.get('iou', 0):.4f}\n\n")
            
            if 'confusion_matrix' in metrics:
                f.write("Confusion Matrix:\n")
                f.write(str(metrics['confusion_matrix']))
                f.write("\n")
        
        print(f"Metrics report saved to {filepath}")


def main():
    """
    Example usage of RandomForestEdgeClassifier
    """
    # Paths
    train_path = "/home/cle/Work/ABC-Challenge/Dataset/Train"
    validation_path = "/home/cle/Work/ABC-Challenge/Dataset/Validation"
    
    # Create data generators
    print("Creating data generators...")
    train_gen = DataGenerator(
        base_path=train_path,
        batch_size=1024,  # Reduced batch size
        shuffle=True,
        target_scales=16,
        target_features=20,
        balance_classes=True,  
        output_format='flat',  # Use flat format for sklearn
        dataset_fraction=0.1  # Use 10% of dataset for faster testing
    )
    
    val_gen = DataGenerator(
        base_path=validation_path,
        batch_size=1024,
        shuffle=False,
        target_scales=16,
        target_features=20,
        balance_classes=True,  
        output_format='flat',
        dataset_fraction=1.0  # Use full validation set
    )
    
    # Initialize classifier with class balancing in RF itself
    rf_classifier = RandomForestEdgeClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Train model - use direct data access instead of batches
    print("Training with direct data access...")
    rf_classifier.train(train_gen, max_batches=None)  # Use all data directly
    
    # Evaluate model
    metrics = rf_classifier.evaluate(val_gen, max_batches=None)
    
    # Get validation data for potential future use
    X_val, y_val = rf_classifier.prepare_data_from_generator(val_gen, max_batches=None)
    
    # Plot feature importance - traditional view
    rf_classifier.plot_feature_importance(top_n=30, filename="gini_importance.png")
    
    # Plot descriptor importance summary - NEW! (What you want)
    rf_classifier.plot_descriptor_importance_summary()
    
    # Export importance data for later analysis
    rf_classifier.export_importance_data()
    
    # Plot feature importance grouped by scale (detailed view)
    rf_classifier.plot_feature_importance_by_scale()
    rf_classifier.plot_heatmap_importance()
    
    # Save model and metrics report
    rf_classifier.save_model()
    rf_classifier.save_metrics_report(metrics)
    
    # Optional: Hyperparameter tuning (comment out for quick testing)
    # best_params = rf_classifier.hyperparameter_tuning(train_gen, max_batches=5, cv=3)
    
    print("Random Forest training and evaluation completed!")
    
    return rf_classifier, metrics


if __name__ == "__main__":
    classifier, results = main()