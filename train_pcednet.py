import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING messages
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Turn off GPU

import tensorflow as tf
import time
import numpy as np

from utils import DataGenerator, analyze_dataset, analyze_full_dataset
from models import PCEDNet
from datetime import datetime



def main(): 
    # Configuration
    # train_path = "/home/cle/Work/ABC-Challenge/Train"
    # val_path = "/home/cle/Work/ABC-Challenge/Validation"

    # IML Testing
    save_path = "/home/cle/data/dtu_results_pc/IML_scan24"
    train_path = "/home/cle/data/dtu_results_pc/IML_scan24"
    val_path = "/home/cle/data/dtu_results_pc/IML_scan24"
    TARGET_SCALES = 16  # Max 16, must be power of 2 to match PCEDNet architecture
    TARGET_FEATURES = 4  # Max 20
    SCALE_SIZE = 15 # 8% // 15% (default) diagonal
    BATCHES = 512
    EPOCHS = 50
    FRACTION = 1 #0.1 (train only 10%) or 1 (train all)
    USE_VALIDATION = False

    print("== Train data generating with balanced batches ==")
    train_generator = DataGenerator(
        train_path, 
        batch_size=BATCHES, 
        shuffle=True,
        target_scales=TARGET_SCALES,
        target_features=TARGET_FEATURES,
        balance_classes=True, 
        num_classes=2,
        dataset_fraction=FRACTION,
        output_format='dict'  # Dictionary format for PCEDNet
    )
    print(f"Training batches: {len(train_generator)}")
    
    val_generator = None
    if USE_VALIDATION:
        print("== Validation data generating with balanced batches ==")
        val_generator = DataGenerator(
            val_path,
            batch_size=BATCHES,
            shuffle=False,
            target_scales=TARGET_SCALES,
            target_features=TARGET_FEATURES,
            balance_classes=True,
            num_classes=2,
            dataset_fraction=FRACTION,
            output_format='dict' # Dictionary format for PCEDNet
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
    #model.save(f"{save_path}/IML_pcednet_{TARGET_SCALES}s_{TARGET_FEATURES}f_{BATCHES}b_{EPOCHS}e_lr01_size{SCALE_SIZE}.h5")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"{save_path}/IML_pcednet_{TARGET_SCALES}s_{TARGET_FEATURES}f_{BATCHES}b_{EPOCHS}e_lr01_size{SCALE_SIZE}_{timestamp}.h5")
    
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