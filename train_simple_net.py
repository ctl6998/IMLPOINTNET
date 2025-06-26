import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING messages
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Turn off GPU

import tensorflow as tf
import time
import numpy as np

from utils import DataGenerator, analyze_dataset
from models import SimpleEdgeNet


def main(): 
    # Configuration
    train_path = "/home/cle/Work/ABC-Challenge/Train"
    val_path = "/home/cle/Work/ABC-Challenge/Validation"
    TARGET_SCALES = 16  # Max 16
    TARGET_FEATURES = 20  # Max 20
    EPOCHS = 5
    FRACTION = 0.1

    print("== Train data generating with balanced batches ==")
    train_generator = DataGenerator(
        train_path, 
        batch_size=2048, 
        shuffle=True,
        target_scales=TARGET_SCALES,
        target_features=TARGET_FEATURES,
        balance_classes=True, 
        num_classes=2,
        dataset_fraction=FRACTION,
        output_format='flat'  # Flattened format for SimpleEdgeNet
    )
    
    print("== Validation data generating with balanced batches ==")
    val_generator = DataGenerator(
        val_path, 
        batch_size=2048, 
        shuffle=False,
        target_scales=TARGET_SCALES,
        target_features=TARGET_FEATURES,
        balance_classes=True,  
        num_classes=2,
        dataset_fraction=FRACTION,
        output_format='flat'  # Flattened format for SimpleEdgeNet
    )

    print("== Analyze batching result ==")
    print(f"Training batches: {len(train_generator)}")  # Total points / batch_size
    print(f"Validation batches: {len(val_generator)}")
    
    # Test a batch to see class distribution
    print("\n== Testing batch balance ==")
    for i in range(min(1, len(train_generator))):  
        sample_inputs, sample_labels = train_generator[i]
        unique, counts = np.unique(sample_labels, return_counts=True)
        print(f"Batch {i} class distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / len(sample_labels)) * 100
            print(f"  Class {cls}: {count} points ({percentage:.2f}%)")
        print(f"  Input shape: {sample_inputs.shape}")  # Should be (batch_size, flattened_features)
        print()
    
    print("\n== Building model ==")
    simple_net = SimpleEdgeNet(
        num_scales=TARGET_SCALES, 
        features_per_scale=TARGET_FEATURES
    )
    model = simple_net.build_model()
    simple_net.compile_model(learning_rate=0.001)

    print("== Model Summary ==")
    simple_net.summary()
    
    print("== Starting training ==")
    history = simple_net.train(
        train_generator=train_generator,
        validation_generator=val_generator,
        epochs=EPOCHS
    )
    
    print("== Save model ==")
    model.save(f"simple_edge_net_{TARGET_SCALES}s_{TARGET_FEATURES}f_{EPOCHS}e.h5")
    
    print("Training completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    # Uncomment to analyze datasets first
    # print("\n=== Dataset Analysis ===")
    # print("\n== Train Dataset Analysis ==")
    # analyze_dataset("/home/cle/Work/ABC-Challenge/Train")
    # print("\n== Validation Dataset Analysis ==")
    # analyze_dataset("/home/cle/Work/ABC-Challenge/Validation")

    print("\n=== Starting Training with Simple Neural Network ===")
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Total training time: {toc - tic:0.4f} seconds")