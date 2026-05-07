
import torch
import os

def load_model(model, path, device):
    """Loads model weights."""
    if os.path.exists(path):
        print(f"Loading model from {path}...")
        model.load_state_dict(torch.load(path, map_location=device))
        return model, True
    else:
        print(f"Model file {path} not found.")
        return model, False

def save_model(model, path):
    """Saves model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
