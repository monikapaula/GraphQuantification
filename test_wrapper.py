import torch
import numpy as np
import sys
import os

from torch_geometric.data import Data
from quantification.wrapper import WrapperClassifier
from splits.presidential_el_split import get_mask

from splits.presidential_el_split import get_mask

# Adjust 'from main' if your load_model function is in a different file
try:
    from main import load_model, MODEL_CONFIG
except ImportError:
    try:
        from train_classifier import load_model, MODEL_CONFIG
    except ImportError:
        print("ERROR: Could not import 'load_model'. Please check where it is defined (main.py?).")
        sys.exit(1)


def test_with_real_model():
    print("\n=== Testing Wrapper with REAL Data and Model ===")

    # 1. Load Real Data
    print("1. Loading Dataset...", end=" ")
    try:
        data, train_mask, val_mask, test_mask = get_mask()
        input_dim = data.x.size(1)
        print(f"OK (Nodes: {data.num_nodes}, Features: {input_dim})")
    except Exception as e:
        print(f"FAILED.\nError: {e}")
        return

    # 2. Load Real Model
    # We try to find the MLP or GCN model you saved earlier
    model_path = "saved_models/mlp_presidential.pth"
    model_name = "MLP"

    if not os.path.exists(model_path):
        # Fallback to GCN if MLP not found
        model_path = "saved_models/GCN_presidential_election.pth"
        model_name = "GCN"
        if not os.path.exists(model_path):
            print(f"\nERROR: No saved model found at 'saved_models/'.\nPlease run 'train_classifier.py' first!")
            return

    print(f"2. Loading Saved Model ({model_name})...", end=" ")
    try:
        # Configure Model
        config = MODEL_CONFIG.copy()
        config['name'] = model_name
        config['input_dim'] = input_dim

        # Instantiate & Load Weights
        model = load_model(config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"OK (Loaded from {model_path})")
    except Exception as e:
        print(f"FAILED.\nError loading model: {e}")
        return

    # 3. Initialize Wrapper
    print("3. Initializing TorchGraphWrapper (This pre-computes probabilities)...", end=" ")
    try:
        # This is where the magic happens. It runs the model on the full graph once.
        wrapper = WrapperClassifier(model, data, device='cpu')
        print("OK")
    except Exception as e:
        print(f"FAILED.\nError initializing wrapper: {e}")
        return

    # 4. Test QuaPy Compatibility
    print("\n--- Running QuaPy Checks ---")

    # Pick 5 random nodes from the VALIDATION set to test
    # (Simulating what QuaPy does during calibration)
    val_indices = np.where(val_mask.cpu().numpy())[0]
    sample_indices = val_indices[:5]
    print(f"Testing on indices: {sample_indices}")

    try:
        # A. Test predict_proba (Crucial)
        probs = wrapper.predict_proba(sample_indices)

        print(f"\n[Check A] .predict_proba() output:")
        print(f"   Shape: {probs.shape} (Expected: (5, 2))")
        print(f"   Type:  {type(probs)} (Expected: <class 'numpy.ndarray'>)")
        print(f"   Sum:   {probs.sum(axis=1)} (Expected: All close to 1.0)")

        if not isinstance(probs, np.ndarray):
            print("   >>> FAIL: Output is not a numpy array.")
            return
        if probs.shape != (5, 2):
            print("   >>> FAIL: Shape mismatch.")
            return

        # B. Test predict (Standard)
        preds = wrapper.predict(sample_indices)
        print(f"\n[Check B] .predict() output:")
        print(f"   Values: {preds}")

        if len(preds) != 5:
            print("   >>> FAIL: Prediction count mismatch.")
            return

        print("\nSUCCESS: The wrapper is working correctly with your real model!")
        print("You can now proceed to run 'evaluate_quantification.py'.")

    except Exception as e:
        print(f"FAILED during prediction checks.\nError: {e}")


if __name__ == "__main__":
    test_with_real_model()

