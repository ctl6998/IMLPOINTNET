#!/usr/bin/env python3
"""
Quick GPU Check for ABC-PCEDNet
Run this to quickly verify if GPU is available for training
"""

def quick_gpu_check():
    print("🔍 Quick GPU Check for ABC-PCEDNet")
    print("=" * 40)
    
    # Check TensorFlow and GPU
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"🖥️  GPUs found: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            
            # Quick computation test
            print("🧪 Testing GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
                c = tf.matmul(a, b)
                print(f"   ✓ GPU computation successful: {c.numpy().flatten()}")
            
            print("🎉 GPU is ready for training!")
            print(f"💡 Recommended settings for your GPU:")
            print(f"   - Batch size: 512-1024")
            print(f"   - Enable mixed precision for better performance")
            return True
        else:
            print("⚠️  No GPU found - will use CPU")
            print("💡 CPU training recommendations:")
            print("   - Batch size: 128-256")
            print("   - Consider using fewer epochs")
            print("   - Training will be slower but still functional")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        print("Run: conda install tensorflow -c conda-forge")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    gpu_available = quick_gpu_check()
    
    print("\n" + "=" * 40)
    if gpu_available:
        print("Status: 🟢 Ready for GPU training")
    else:
        print("Status: 🟡 CPU training only")
    print("You can now run: python train_pcednet.py")