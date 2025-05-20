import os
import subprocess
from pathlib import Path

def check_and_train_models():
    classic_model_exists = os.path.exists('model_classic.joblib')
    modern_model_exists = os.path.exists('model_modern.pak')

    if not classic_model_exists:
        print("Classic model not found. Training classic model...")
        try:
            subprocess.run(['python', '-m', 'setup/train_classic.py'], check=True)
            print("Classic model training completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error training classic model: {e}")
            return False

    if not modern_model_exists:
        print("Modern model not found. Training modern model...")
        try:
            subprocess.run(['python', '-m', 'setup/train_modern.py'], check=True)
            print("Modern model training completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error training modern model: {e}")
            return False

    return True

def main():
    if not check_and_train_models():
        print("Failed to train required models. Exiting...")
        return

    print("All models are ready. Proceeding with main execution...")

if __name__ == "__main__":
    main()
