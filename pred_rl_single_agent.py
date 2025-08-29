import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from datetime import datetime
from utils import DataGenerator

# Configuration
TARGET_SCALES = 16
TARGET_FEATURES = 20
FRACTION = 0.1  # Reduce to 10% for faster training
n_descriptors = TARGET_FEATURES  # 20 descriptors (grouped across scales)
state_size = 49
action_size = 2 * n_descriptors + 1  # 41 actions: 20 add + 20 remove + 1 stop

# Performance optimization flags
FAST_MODE = True  # Enable fast training mode
RF_ESTIMATORS = 20 if FAST_MODE else 50  # Fewer trees for speed
BATCH_TRAIN_INTERVAL = 5 if FAST_MODE else 1  # Train DQN less frequently

# Global tracking for descriptor rewards
descriptor_rewards = np.zeros(n_descriptors)  # Track cumulative reward per descriptor
descriptor_counts = np.zeros(n_descriptors)   # Track how many times each descriptor was used

# Create output directory and setup logging
output_dir = "result_rl_single_agent"
os.makedirs(output_dir, exist_ok=True)

# Setup logging to capture all printed output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Start logging
log_file = os.path.join(output_dir, f'rl_feature_selection_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
sys.stdout = Logger(log_file)

# Set paths
train_path = "/home/cle/Work/ABC-Challenge/Dataset/Train"
val_path = "/home/cle/Work/ABC-Challenge/Dataset/Validation"

# Load data using DataGenerator
print("Loading training data...")
train_generator = DataGenerator(
    train_path, 
    batch_size=2048, 
    shuffle=True,
    target_scales=TARGET_SCALES,
    target_features=TARGET_FEATURES,
    balance_classes=True, 
    num_classes=2,
    dataset_fraction=FRACTION,
    output_format='flat'
)

val_generator = DataGenerator(
    val_path,
    batch_size=2048,
    shuffle=False,
    target_scales=TARGET_SCALES, 
    target_features=TARGET_FEATURES,
    balance_classes=False,
    num_classes=2,
    dataset_fraction=1.0,
    output_format='flat'
)

# Get all training data
X_train_list = []
y_train_list = []
for i in range(len(train_generator)):
    X_batch, y_batch = train_generator[i]
    X_train_list.append(X_batch)
    y_train_list.append(y_batch.flatten())

X_train = np.vstack(X_train_list)
y_train = np.hstack(y_train_list)

# Get all validation data
X_val_list = []
y_val_list = []
for i in range(len(val_generator)):
    X_batch, y_batch = val_generator[i]
    X_val_list.append(X_batch)
    y_val_list.append(y_batch.flatten())

X_val = np.vstack(X_val_list)
y_val = np.hstack(y_val_list)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Training labels distribution: {np.bincount(y_train.astype(int))}")

# Group features by descriptors (sum across 16 scales)
print("Grouping features by descriptors...")
X_train_grouped = np.zeros((X_train.shape[0], n_descriptors))
X_val_grouped = np.zeros((X_val.shape[0], n_descriptors))

for desc in range(n_descriptors):
    # Sum descriptor across all 16 scales
    desc_indices = [desc + scale * TARGET_FEATURES for scale in range(TARGET_SCALES)]
    X_train_grouped[:, desc] = np.sum(X_train[:, desc_indices], axis=1)
    X_val_grouped[:, desc] = np.sum(X_val[:, desc_indices], axis=1)

X_train = X_train_grouped
X_val = X_val_grouped

print(f"After grouping - Training data shape: {X_train.shape}")
print(f"After grouping - Validation data shape: {X_val.shape}")

# Build DQN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=state_size, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

target_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=state_size, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])
target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Copy weights
target_model.set_weights(model.get_weights())

# RL parameters
memory = deque(maxlen=2000)
gamma = 0.95
min_epsilon = 0.1  # Minimum 10% exploration
episodes = 5  # Total number of episodes

# Function to calculate state (49-length vector)
def get_state(selected_features):
    if len(selected_features) == 0:
        return np.zeros(49)
    
    # Get selected features data
    selected_indices = list(selected_features)
    selected_data = X_train[:, selected_indices]
    
    # Calculate 7 stats for each selected feature
    feature_stats = []
    for i in range(len(selected_indices)):
        feature = selected_data[:, i]
        stats = [
            np.std(feature),
            np.mean(feature),
            np.min(feature),
            np.max(feature),
            np.percentile(feature, 25),
            np.percentile(feature, 50),
            np.percentile(feature, 75)
        ]
        feature_stats.extend(stats)
    
    # Pad or truncate to 49 values  
    if len(feature_stats) > 49:
        feature_stats = feature_stats[:49]
    else:
        feature_stats.extend([0] * (49 - len(feature_stats)))
    
    # Calculate stats across stats (7x7=49)
    stats_array = np.array(feature_stats).reshape(-1, 7)
    final_stats = []
    for i in range(7):
        col = stats_array[:, i]
        if len(col) > 0:
            col_stats = [
                np.std(col),
                np.mean(col), 
                np.min(col),
                np.max(col),
                np.percentile(col, 25),
                np.percentile(col, 50),
                np.percentile(col, 75)
            ]
            final_stats.extend(col_stats)
    
    return np.array(final_stats[:49])

# Function to calculate reward and update descriptor rewards
def calculate_reward(selected_features):
    if len(selected_features) == 0:
        return 0
    
    selected_indices = list(selected_features)
    X_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    
    # Train Random Forest (main bottleneck - reduce trees)
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, n_jobs=-1)
    rf.fit(X_selected, y_train)
    
    # Calculate accuracy
    y_pred = rf.predict(X_val_selected)
    accuracy = accuracy_score(y_val, y_pred)
    
    # Calculate relevance
    # Paper formula: Rv = (1/|S|) * Σ I(xi; c)
    mi_scores = mutual_info_classif(X_selected, y_train, random_state=42)
    relevance = np.sum(mi_scores) / len(selected_indices)
    
    # Calculate redundancy (mutual information between features)
    # Paper formula: Rd = (1/|S|²) * Σ I(xi; xj)
    redundancy = 0
    if len(selected_indices) > 1:
        redundancy_scores = []
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                mi = mutual_info_regression(X_selected[:, [i]], X_selected[:, j], random_state=42)
                redundancy_scores.append(mi[0])
        redundancy = np.sum(redundancy_scores) / (len(selected_indices) ** 2) if redundancy_scores else 0

    print(f"Accuracy: {accuracy:.4f}, Relevance: {relevance:.4f}, Redundancy: {redundancy:.4f}")
    total_reward = accuracy + relevance - 0.1 * redundancy
    
    # Distribute reward equally among selected descriptors
    reward_per_descriptor = total_reward / len(selected_features)
    
    # Update global descriptor rewards
    for desc_idx in selected_indices:
        descriptor_rewards[desc_idx] += reward_per_descriptor
        descriptor_counts[desc_idx] += 1
    
    return total_reward

# Function to get valid actions
def get_valid_actions(selected_features, max_features=15):
    valid_actions = []
    
    # Add actions (0 to 19)
    for i in range(n_descriptors):
        if i not in selected_features and len(selected_features) < max_features:
            valid_actions.append(i)
    
    # Remove actions (20 to 39)  
    for i in range(n_descriptors):
        if i in selected_features:
            valid_actions.append(n_descriptors + i)
    
    # Stop action (40)
    valid_actions.append(2 * n_descriptors)
    
    return valid_actions

# Function for agent action selection
def select_action(state, valid_actions, epsilon):
    if np.random.random() <= epsilon:
        return random.choice(valid_actions)
    
    q_values = model.predict(state.reshape(1, -1), verbose=0)
    valid_q_values = [(action, q_values[0][action]) for action in valid_actions]
    return max(valid_q_values, key=lambda x: x[1])[0]

def create_descriptor_reward_graph(avg_descriptor_rewards):
    """Create and save column graph of descriptor rewards"""
    
    # Create output directory if it doesn't exist
    output_dir = "result_rl_single_agent"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the column graph
    fig, ax = plt.subplots(figsize=(14, 8))
    
    descriptor_indices = np.arange(n_descriptors)
    bars = ax.bar(descriptor_indices, avg_descriptor_rewards, 
                  color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Descriptor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('RL Feature Selection - Descriptor Importance\n(Total importance across all experiments)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis ticks to show all descriptor indices
    ax.set_xticks(descriptor_indices)
    ax.set_xticklabels([f'Desc{i+1}' for i in descriptor_indices])
    
    # Add value labels on top of bars
    for i, (bar, reward) in enumerate(zip(bars, avg_descriptor_rewards)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{reward:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(avg_descriptor_rewards) * 1.15 if max(avg_descriptor_rewards) > 0 else 1)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the graph
    graph_path = os.path.join(output_dir, 'descriptor_rewards_column_graph.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'descriptor_rewards_column_graph.pdf'), bbox_inches='tight')
    
    print(f"\nColumn graph saved to: {graph_path}")
    print(f"PDF version also saved: {os.path.join(output_dir, 'descriptor_rewards_column_graph.pdf')}")
    
    # Show the plot (optional - comment out if running headless)
    plt.show()
    plt.close()

# Training loop
best_reward = -np.inf
best_features = set()
scores = []

print("Starting RL training...")
print("=" * 60)

for episode in range(episodes):
    print(f"\n--- Episode {episode + 1}/{episodes} ---")
    
    # Linear decay: epsilon goes from 1.0 to 0.1 over all episodes
    epsilon = 1.0 - (episode / (episodes - 1)) * (1.0 - min_epsilon)
    
    # Calculate exploration/exploitation percentages for this episode
    percentage_random = epsilon * 100
    percentage_dqn = (1 - epsilon) * 100
    print(f"Episode {episode + 1} - Epsilon: {epsilon:.3f} ({percentage_random:.1f}% random, {percentage_dqn:.1f}% DQN)")
    
    # Reset environment
    selected_features = set()
    total_reward = 0
    steps = 0
    max_steps = 20
    print(f"Starting with empty feature set...")
    
    # Get initial state
    state = get_state(selected_features)
    print(f"Initial state shape: {state.shape}")
    
    # Episode loop
    while steps < max_steps:
        # Get valid actions
        valid_actions = get_valid_actions(selected_features)
        
        print(f"Step {steps}: Currently selected descriptors: {sorted(selected_features) if selected_features else 'None'}")
        
        # Select action
        action = select_action(state, valid_actions, epsilon)
        
        action_type = ""
        if action < n_descriptors:
            action_type = f"ADD descriptor {action}"
        elif action < 2 * n_descriptors:
            feature_idx = action - n_descriptors
            action_type = f"REMOVE descriptor {feature_idx}"
        else:
            action_type = "STOP"
        
        print(f"  Action chosen: {action_type}")
        
        # Take action
        if action < n_descriptors:
            # Add descriptor
            selected_features.add(action)
        elif action < 2 * n_descriptors:
            # Remove descriptor
            feature_to_remove = action - n_descriptors
            selected_features.discard(feature_to_remove)
        else:
            # Stop action
            print("  Agent chose to STOP")
            break
        
        # Get reward and next state
        print(f"  New selection: {sorted(selected_features)}")
        print(f"  Calculating reward for descriptors: {sorted(selected_features)}")
        reward = calculate_reward(selected_features)
        next_state = get_state(selected_features)
        reward_per_desc = reward / len(selected_features) if len(selected_features) > 0 else 0
        print(f"  Total reward: {reward:.4f}, Per descriptor: {reward_per_desc:.4f}")
        
        # Store experience
        done = (action == 2 * n_descriptors)
        memory.append((state, action, reward, next_state, done))
        
        total_reward += reward
        state = next_state
        steps += 1
        
        # Stop if done
        if done or len(selected_features) >= 15:
            if len(selected_features) >= 15:
                print("  Reached maximum descriptors limit")
            break
    
    # Train model if enough experiences
    if len(memory) >= 32:
        print(f"  Training DQN with {len(memory)} experiences...")
        # Sample batch
        batch = random.sample(memory, 32)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Calculate Q-values
        current_q_values = model.predict(states, verbose=0)
        next_q_values = target_model.predict(next_states, verbose=0)
        
        targets = current_q_values.copy()
        
        # Apply Bellman equation
        for i in range(32):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
        
        # Train model
        loss = model.fit(states, targets, epochs=1, verbose=0)
        print(f"  DQN training loss: {loss.history['loss'][0]:.6f}")
    
    # Update target network every 10 episodes
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())
        print(f"  Updated target network weights")
    
    # Track best features
    if total_reward > best_reward:
        best_reward = total_reward
        best_features = selected_features.copy()
        print(f"  NEW BEST! Reward: {best_reward:.4f} with {len(best_features)} descriptors")
    
    scores.append(total_reward)
    
    # Episode summary
    avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
    print(f"Episode {episode + 1} Summary:")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Steps taken: {steps}")
    print(f"  Final descriptors: {len(selected_features)}")
    print(f"  Epsilon: {epsilon:.3f} ({percentage_random:.1f}% random, {percentage_dqn:.1f}% DQN)")
    print(f"  Avg last 10 episodes: {avg_score:.4f}")
    print(f"  Best so far: {best_reward:.4f} ({len(best_features)} descriptors)")

# Final evaluation
print(f"\nTraining completed!")
print(f"Best descriptors found: {sorted(best_features)}")
print(f"Number of best descriptors: {len(best_features)}")

# Evaluate final performance
if len(best_features) > 0:
    selected_indices = list(best_features)
    X_train_final = X_train[:, selected_indices]
    X_val_final = X_val[:, selected_indices]
    
    # Train final Random Forest
    final_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_rf.fit(X_train_final, y_train)
    
    # Get final accuracy
    y_pred_final = final_rf.predict(X_val_final)
    final_accuracy = accuracy_score(y_val, y_pred_final)
    
    print(f"Final validation accuracy: {final_accuracy:.4f}")
    print(f"Selected descriptor indices: {selected_indices}")
else:
    print("No descriptors were selected!")

# Generate final descriptor ranking
print(f"\n{'='*80}")
print("DESCRIPTOR RANKING BY USAGE")
print(f"{'='*80}")

# Sort descriptors by times used (frequency) 
frequency_ranking = [(i, descriptor_rewards[i], descriptor_counts[i]) for i in range(n_descriptors)]
frequency_ranking.sort(key=lambda x: x[2], reverse=True)  # Sort by times used (index 2)

print("Rank | Descriptor | Times Used | Total Reward")
print("-" * 47)
for rank, (desc_idx, total_reward, count) in enumerate(frequency_ranking):
    print(f"{rank+1:4d} | Desc{desc_idx+1:2d}     | {count:9.0f}  | {total_reward:11.4f}")

print(f"\nTop Descriptors by Usage Frequency:")
print(f"Top 4 most used: {[f'Desc{frequency_ranking[i][0]+1}' for i in range(min(4, len(frequency_ranking)))]}")
print(f"Top 8 most used: {[f'Desc{frequency_ranking[i][0]+1}' for i in range(min(8, len(frequency_ranking)))]}")
print(f"Top 12 most used: {[f'Desc{frequency_ranking[i][0]+1}' for i in range(min(12, len(frequency_ranking)))]}")

# Create and save the column graph (using times used instead of average reward)
descriptor_usage_counts = descriptor_counts
create_descriptor_reward_graph(descriptor_usage_counts)

# import numpy as np
# import tensorflow as tf
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
# from collections import deque
# import random
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# import sys
# from datetime import datetime
# from utils import DataGenerator

# # Configuration
# TARGET_SCALES = 16
# TARGET_FEATURES = 20
# FRACTION = 0.1  # Reduce to 10% for faster training
# n_descriptors = TARGET_FEATURES  # 20 descriptors (grouped across scales)
# state_size = 49
# action_size = 2 * n_descriptors + 1  # 41 actions: 20 add + 20 remove + 1 stop

# # Performance optimization flags
# FAST_MODE = True  # Enable fast training mode
# RF_ESTIMATORS = 20 if FAST_MODE else 50  # Fewer trees for speed
# BATCH_TRAIN_INTERVAL = 5 if FAST_MODE else 1  # Train DQN less frequently

# # Global tracking for descriptor rewards
# descriptor_rewards = np.zeros(n_descriptors)  # Track cumulative reward per descriptor
# descriptor_counts = np.zeros(n_descriptors)   # Track how many times each descriptor was used

# # Create output directory and setup logging
# output_dir = "result_rl_single_agent"
# os.makedirs(output_dir, exist_ok=True)

# # Setup logging to capture all printed output
# class Logger:
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         self.log = open(filename, 'w', encoding='utf-8')
        
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
        
#     def flush(self):
#         self.terminal.flush()
#         self.log.flush()

# # Start logging
# log_file = os.path.join(output_dir, f'rl_feature_selection_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
# sys.stdout = Logger(log_file)

# # Set paths
# train_path = "/home/cle/Work/ABC-Challenge/Dataset/Train"
# val_path = "/home/cle/Work/ABC-Challenge/Dataset/Validation"

# # Load data using DataGenerator
# print("Loading training data...")
# train_generator = DataGenerator(
#     train_path, 
#     batch_size=2048, 
#     shuffle=True,
#     target_scales=TARGET_SCALES,
#     target_features=TARGET_FEATURES,
#     balance_classes=True, 
#     num_classes=2,
#     dataset_fraction=FRACTION,
#     output_format='flat'
# )

# val_generator = DataGenerator(
#     val_path,
#     batch_size=2048,
#     shuffle=False,
#     target_scales=TARGET_SCALES, 
#     target_features=TARGET_FEATURES,
#     balance_classes=False,
#     num_classes=2,
#     dataset_fraction=1.0,
#     output_format='flat'
# )

# # Get all training data
# X_train_list = []
# y_train_list = []
# for i in range(len(train_generator)):
#     X_batch, y_batch = train_generator[i]
#     X_train_list.append(X_batch)
#     y_train_list.append(y_batch.flatten())

# X_train = np.vstack(X_train_list)
# y_train = np.hstack(y_train_list)

# # Get all validation data
# X_val_list = []
# y_val_list = []
# for i in range(len(val_generator)):
#     X_batch, y_batch = val_generator[i]
#     X_val_list.append(X_batch)
#     y_val_list.append(y_batch.flatten())

# X_val = np.vstack(X_val_list)
# y_val = np.hstack(y_val_list)

# print(f"Training data shape: {X_train.shape}")
# print(f"Validation data shape: {X_val.shape}")
# print(f"Training labels distribution: {np.bincount(y_train.astype(int))}")

# # Group features by descriptors (sum across 16 scales)
# print("Grouping features by descriptors...")
# X_train_grouped = np.zeros((X_train.shape[0], n_descriptors))
# X_val_grouped = np.zeros((X_val.shape[0], n_descriptors))

# for desc in range(n_descriptors):
#     # Sum descriptor across all 16 scales
#     desc_indices = [desc + scale * TARGET_FEATURES for scale in range(TARGET_SCALES)]
#     X_train_grouped[:, desc] = np.sum(X_train[:, desc_indices], axis=1)
#     X_val_grouped[:, desc] = np.sum(X_val[:, desc_indices], axis=1)

# X_train = X_train_grouped
# X_val = X_val_grouped

# print(f"After grouping - Training data shape: {X_train.shape}")
# print(f"After grouping - Validation data shape: {X_val.shape}")

# # Build DQN model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, input_dim=state_size, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(action_size, activation='linear')
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# target_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, input_dim=state_size, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(action_size, activation='linear')
# ])
# target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# # Copy weights
# target_model.set_weights(model.get_weights())

# # RL parameters
# memory = deque(maxlen=2000)
# epsilon = 1.0
# epsilon_min = 0.01
# epsilon_decay = 0.995
# gamma = 0.95

# # Function to calculate state (49-length vector)
# def get_state(selected_features):
#     if len(selected_features) == 0:
#         return np.zeros(49)
    
#     # Get selected features data
#     selected_indices = list(selected_features)
#     selected_data = X_train[:, selected_indices]
    
#     # Calculate 7 stats for each selected feature
#     feature_stats = []
#     for i in range(len(selected_indices)):
#         feature = selected_data[:, i]
#         stats = [
#             np.std(feature),
#             np.mean(feature),
#             np.min(feature),
#             np.max(feature),
#             np.percentile(feature, 25),
#             np.percentile(feature, 50),
#             np.percentile(feature, 75)
#         ]
#         feature_stats.extend(stats)
    
#     # Pad or truncate to 49 values  
#     if len(feature_stats) > 49:
#         feature_stats = feature_stats[:49]
#     else:
#         feature_stats.extend([0] * (49 - len(feature_stats)))
    
#     # Calculate stats across stats (7x7=49)
#     stats_array = np.array(feature_stats).reshape(-1, 7)
#     final_stats = []
#     for i in range(7):
#         col = stats_array[:, i]
#         if len(col) > 0:
#             col_stats = [
#                 np.std(col),
#                 np.mean(col), 
#                 np.min(col),
#                 np.max(col),
#                 np.percentile(col, 25),
#                 np.percentile(col, 50),
#                 np.percentile(col, 75)
#             ]
#             final_stats.extend(col_stats)
    
#     return np.array(final_stats[:49])

# # Function to calculate reward and update descriptor rewards
# def calculate_reward(selected_features):
#     if len(selected_features) == 0:
#         return 0
    
#     selected_indices = list(selected_features)
#     X_selected = X_train[:, selected_indices]
#     X_val_selected = X_val[:, selected_indices]
    
#     # Train Random Forest (main bottleneck - reduce trees)
#     rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=42, n_jobs=-1)
#     rf.fit(X_selected, y_train)
    
#     # Calculate accuracy
#     y_pred = rf.predict(X_val_selected)
#     accuracy = accuracy_score(y_val, y_pred)
    
#     # Calculate relevance
#     # Paper formula: Rv = (1/|S|) * Σ I(xi; c)
#     mi_scores = mutual_info_classif(X_selected, y_train, random_state=42)
#     relevance = np.sum(mi_scores) / len(selected_indices)
#     # relevance = np.mean(mi_scores) 
    
#     # Calculate redundancy (mutual information between features)
#     # Paper formula: Rd = (1/|S|²) * Σ I(xi; xj)
#     redundancy = 0
#     if len(selected_indices) > 1:
#         redundancy_scores = []
#         for i in range(len(selected_indices)):
#             for j in range(i+1, len(selected_indices)):
#                 mi = mutual_info_regression(X_selected[:, [i]], X_selected[:, j], random_state=42)
#                 redundancy_scores.append(mi[0])
#         redundancy = np.sum(redundancy_scores) / (len(selected_indices) ** 2) if redundancy_scores else 0

#     print(f"Accuracy: {accuracy:.4f}, Relevance: {relevance:.4f}, Redundancy: {redundancy:.4f}")
#     total_reward = accuracy + relevance - 0.1 * redundancy
#     # total_reward = accuracy
    
#     # Distribute reward equally among selected descriptors
#     reward_per_descriptor = total_reward / len(selected_features)
    
#     # Update global descriptor rewards
#     for desc_idx in selected_indices:
#         descriptor_rewards[desc_idx] += reward_per_descriptor
#         descriptor_counts[desc_idx] += 1
    
#     return total_reward

# # Function to get valid actions
# def get_valid_actions(selected_features, max_features=15):
#     valid_actions = []
    
#     # Add actions (0 to 19)
#     for i in range(n_descriptors):
#         if i not in selected_features and len(selected_features) < max_features:
#             valid_actions.append(i)
    
#     # Remove actions (20 to 39)  
#     for i in range(n_descriptors):
#         if i in selected_features:
#             valid_actions.append(n_descriptors + i)
    
#     # Stop action (40)
#     valid_actions.append(2 * n_descriptors)
    
#     return valid_actions

# # Function for agent action selection
# def select_action(state, valid_actions, epsilon):
#     if np.random.random() <= epsilon:
#         return random.choice(valid_actions)
    
#     q_values = model.predict(state.reshape(1, -1), verbose=0)
#     valid_q_values = [(action, q_values[0][action]) for action in valid_actions]
#     return max(valid_q_values, key=lambda x: x[1])[0]

# def create_descriptor_reward_graph(avg_descriptor_rewards):
#     """Create and save column graph of descriptor rewards"""
    
#     # Create output directory if it doesn't exist
#     output_dir = "result_rl_single_agent"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create the column graph
#     fig, ax = plt.subplots(figsize=(14, 8))
    
#     descriptor_indices = np.arange(n_descriptors)
#     bars = ax.bar(descriptor_indices, avg_descriptor_rewards, 
#                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
#     # Customize the plot
#     ax.set_xlabel('Descriptor', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
#     ax.set_title('RL Feature Selection - Descriptor Importance\n(Total importance across all experiments)', 
#                  fontsize=14, fontweight='bold', pad=20)
    
#     # Set x-axis ticks to show all descriptor indices
#     ax.set_xticks(descriptor_indices)
#     ax.set_xticklabels([f'Desc{i+1}' for i in descriptor_indices])
    
#     # Add value labels on top of bars
#     for i, (bar, reward) in enumerate(zip(bars, avg_descriptor_rewards)):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
#                 f'{reward:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
#     # Set y-axis to start from 0
#     ax.set_ylim(0, max(avg_descriptor_rewards) * 1.15 if max(avg_descriptor_rewards) > 0 else 1)
    
#     # Remove top and right spines for cleaner look
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save the graph
#     graph_path = os.path.join(output_dir, 'descriptor_rewards_column_graph.png')
#     plt.savefig(graph_path, dpi=300, bbox_inches='tight')
#     plt.savefig(os.path.join(output_dir, 'descriptor_rewards_column_graph.pdf'), bbox_inches='tight')
    
#     print(f"\nColumn graph saved to: {graph_path}")
#     print(f"PDF version also saved: {os.path.join(output_dir, 'descriptor_rewards_column_graph.pdf')}")
    
#     # Show the plot (optional - comment out if running headless)
#     plt.show()
#     plt.close()

# # Training loop
# episodes = 2
# best_reward = -np.inf
# best_features = set()
# scores = []

# print("Starting RL training...")
# print("=" * 60)

# for episode in range(episodes):
#     print(f"\n--- Episode {episode + 1}/{episodes} ---")
    
#     # Calculate exploration/exploitation percentages for this episode
#     percentage_random = epsilon * 100
#     percentage_dqn = (1 - epsilon) * 100
#     print(f"Episode {episode + 1} - Epsilon: {epsilon:.3f} ({percentage_random:.1f}% random, {percentage_dqn:.1f}% DQN)")
    
#     # Reset environment
#     selected_features = set()
#     total_reward = 0
#     steps = 0
#     max_steps = 10
#     print(f"Starting with empty feature set...")
    
#     # Get initial state
#     state = get_state(selected_features)
#     print(f"Initial state shape: {state.shape}")
    
#     # Episode loop
#     while steps < max_steps:
#         # Get valid actions
#         valid_actions = get_valid_actions(selected_features)
        
#         print(f"Step {steps}: Currently selected descriptors: {sorted(selected_features) if selected_features else 'None'}")
        
#         # Select action
#         action = select_action(state, valid_actions, epsilon)
        
#         action_type = ""
#         if action < n_descriptors:
#             action_type = f"ADD descriptor {action}"
#         elif action < 2 * n_descriptors:
#             feature_idx = action - n_descriptors
#             action_type = f"REMOVE descriptor {feature_idx}"
#         else:
#             action_type = "STOP"
        
#         print(f"  Action chosen: {action_type}")
        
#         # Take action
#         if action < n_descriptors:
#             # Add descriptor
#             selected_features.add(action)
#         elif action < 2 * n_descriptors:
#             # Remove descriptor
#             feature_to_remove = action - n_descriptors
#             selected_features.discard(feature_to_remove)
#         else:
#             # Stop action
#             print("  Agent chose to STOP")
#             break
        
#         # Get reward and next state
#         print(f"  New selection: {sorted(selected_features)}")
#         print(f"  Calculating reward for descriptors: {sorted(selected_features)}")
#         reward = calculate_reward(selected_features)
#         next_state = get_state(selected_features)
#         reward_per_desc = reward / len(selected_features) if len(selected_features) > 0 else 0
#         print(f"  Total reward: {reward:.4f}, Per descriptor: {reward_per_desc:.4f}")
        
#         # Store experience
#         done = (action == 2 * n_descriptors)
#         memory.append((state, action, reward, next_state, done))
        
#         total_reward += reward
#         state = next_state
#         steps += 1
        
#         # Stop if done
#         if done or len(selected_features) >= 15:
#             if len(selected_features) >= 15:
#                 print("  Reached maximum descriptors limit")
#             break
    
#     # Train model if enough experiences
#     if len(memory) >= 32:
#         print(f"  Training DQN with {len(memory)} experiences...")
#         # Sample batch
#         batch = random.sample(memory, 32) #Learn from 32 experience at once
#         states = np.array([e[0] for e in batch])
#         actions = np.array([e[1] for e in batch])
#         rewards = np.array([e[2] for e in batch])
#         next_states = np.array([e[3] for e in batch])
#         dones = np.array([e[4] for e in batch])
        
#         # Calculate Q-values for actions of states in memory experiences
#         current_q_values = model.predict(states, verbose=0) # Q-values of current states (using DQN)
#         next_q_values = target_model.predict(next_states, verbose=0) # Q-value for next state (using DQN)

#         # Using loss between Q-values get from experiences (in experiences it derives from random or DQN of current state)
#         # and Q-values get from  next state (using DQN)
#         ##### Example
#         # Given experiment:
#         ## State: [0.5, 0.3,...] (current feature stats)
#         ## Action: 1 (remove feature)
#         ## Reward: 0.8 (actual reward received)
#         ## Next State: [0.4, 0.2] (new feature stats after action)
#         ## Done: False (episode continues)
#         ## Gamma: 0.9 (discount factor)

#         # DQN(current_state): [0.5, 0.3,...] -> Q-values of 21 action of current state: add/remove/stop [0.2, 0.5, 0.3] => This is what we actually get*
#         # DQN(next_state) [0.4, 0.2,..] -> Q_values of 21 action of next steat: add/remove/stop [0.1, 0.6, 0.4] => This is what we actually get

#         # Expected Q-values from Bellman equation (given curren + next state, the Q_values of moving from current to next suppose to be:)
#         ## targets = [0.2, 0.5, 0.3]
#         ## targets[1] = reward + gamma * max(next_q_values)
#         ## targets[1] = 0.8 + 0.9 * max([0.1, 0.6, 0.4]) = 1.34
#         ## ... similar, target [0.2, 1.34, 0.3] ==> This is the Q-values it suppose to get*

#         # Train network to predict [0.2, 1.34, 0.3] when given state [0.5, 0.3,...], current we get [0.2, 0.5, 0.3]
#         # model.fit([0.5, 0.3], [0.2, 1.34, 0.3])
        
#         targets = current_q_values.copy()

#         # Apply Bellman equation for each experience
#         for i in range(32):
#             if dones[i]:
#                 targets[i][actions[i]] = rewards[i]
#             else:
#                 targets[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
        
#         # Train model
#         loss = model.fit(states, targets, epochs=1, verbose=0)
#         print(f"  DQN training loss: {loss.history['loss'][0]:.6f}")
    
#     # Update target network every 10 episodes
#     if episode % 10 == 0:
#         target_model.set_weights(model.get_weights()) # Update weight of target_model slower
#         # This ensure that the Q-values from current state increase "faster" than the target (using Q-values from next state)
#         print(f"  Updated target network weights")
    
#     # Update epsilon
#     if epsilon > epsilon_min:
#         epsilon *= epsilon_decay
    
#     # Track best features
#     if total_reward > best_reward:
#         best_reward = total_reward
#         best_features = selected_features.copy()
#         print(f"  NEW BEST! Reward: {best_reward:.4f} with {len(best_features)} descriptors")
    
#     scores.append(total_reward)
    
#     # Episode summary
#     avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
#     final_percentage_random = epsilon * 100
#     final_percentage_dqn = (1 - epsilon) * 100
#     print(f"Episode {episode + 1} Summary:")
#     print(f"  Total reward: {total_reward:.4f}")
#     print(f"  Steps taken: {steps}")
#     print(f"  Final descriptors: {len(selected_features)}")
#     print(f"  Epsilon: {epsilon:.3f} ({final_percentage_random:.1f}% random, {final_percentage_dqn:.1f}% DQN)")
#     print(f"  Avg last 10 episodes: {avg_score:.4f}")
#     print(f"  Best so far: {best_reward:.4f} ({len(best_features)} descriptors)")

# # Final evaluation
# print(f"\nTraining completed!")
# print(f"Best descriptors found: {sorted(best_features)}")
# print(f"Number of best descriptors: {len(best_features)}")

# # Evaluate final performance
# if len(best_features) > 0:
#     selected_indices = list(best_features)
#     X_train_final = X_train[:, selected_indices]
#     X_val_final = X_val[:, selected_indices]
    
#     # Train final Random Forest
#     final_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     final_rf.fit(X_train_final, y_train)
    
#     # Get final accuracy
#     y_pred_final = final_rf.predict(X_val_final)
#     final_accuracy = accuracy_score(y_val, y_pred_final)
    
#     print(f"Final validation accuracy: {final_accuracy:.4f}")
#     print(f"Selected descriptor indices: {selected_indices}")
# else:
#     print("No descriptors were selected!")

# # Generate final descriptor ranking
# print(f"\n{'='*80}")
# print("DESCRIPTOR RANKING BY USAGE")
# print(f"{'='*80}")

# # Sort descriptors by times used (frequency) 
# frequency_ranking = [(i, descriptor_rewards[i], descriptor_counts[i]) for i in range(n_descriptors)]
# frequency_ranking.sort(key=lambda x: x[2], reverse=True)  # Sort by times used (index 2)

# print("Rank | Descriptor | Times Used | Total Reward")
# print("-" * 47)
# for rank, (desc_idx, total_reward, count) in enumerate(frequency_ranking):
#     print(f"{rank+1:4d} | Desc{desc_idx+1:2d}     | {count:9.0f}  | {total_reward:11.4f}")

# print(f"\nTop Descriptors by Usage Frequency:")
# print(f"Top 4 most used: {[f'Desc{frequency_ranking[i][0]+1}' for i in range(min(4, len(frequency_ranking)))]}")
# print(f"Top 8 most used: {[f'Desc{frequency_ranking[i][0]+1}' for i in range(min(8, len(frequency_ranking)))]}")
# print(f"Top 12 most used: {[f'Desc{frequency_ranking[i][0]+1}' for i in range(min(12, len(frequency_ranking)))]}")

# # Create and save the column graph (using times used instead of average reward)
# descriptor_usage_counts = descriptor_counts
# create_descriptor_reward_graph(descriptor_usage_counts)