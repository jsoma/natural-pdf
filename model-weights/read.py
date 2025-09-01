import json
from pathlib import Path

import torch


def inspect_rfdetr_nano_model(model_path):
    """
    Inspect RF-DETR Nano model and extract configuration details
    """
    print(f"Loading model from: {model_path}")

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Check the structure of the checkpoint
    if isinstance(checkpoint, dict):
        print("\n=== Checkpoint Keys ===")
        for key in checkpoint.keys():
            print(f"- {key}")
            if key == "model" and hasattr(checkpoint[key], "state_dict"):
                print("  (contains state_dict)")

        # Look for model configuration
        if "config" in checkpoint:
            print("\n=== Model Config Found ===")
            config = checkpoint["config"]
            print(json.dumps(config, indent=2))

        # Look for class information
        if "names" in checkpoint:
            print("\n=== Class Names Found ===")
            class_names = checkpoint["names"]
            print(f"Number of classes: {len(class_names)}")
            print("Classes:", class_names)

        # Check model state dict for architecture hints
        if "model" in checkpoint:
            model_data = checkpoint["model"]
            if hasattr(model_data, "state_dict"):
                state_dict = model_data.state_dict()
            elif isinstance(model_data, dict):
                state_dict = model_data
            else:
                state_dict = {}

            if state_dict:
                print("\n=== Model Architecture Hints ===")
                # Look for the final classification layer to determine number of classes
                for key in state_dict.keys():
                    if "class_embed" in key or "cls_head" in key or "num_classes" in key:
                        print(f"{key}: shape = {state_dict[key].shape}")
                        if len(state_dict[key].shape) > 0:
                            # The last dimension often indicates number of classes
                            potential_num_classes = state_dict[key].shape[-1]
                            print(f"  Potential number of classes: {potential_num_classes}")

                # Check for bbox embed layers
                for key in state_dict.keys():
                    if "bbox_embed" in key or "box_head" in key:
                        print(f"{key}: shape = {state_dict[key].shape}")
                        break

        # Check for Roboflow-specific metadata
        if "roboflow" in checkpoint:
            print("\n=== Roboflow Metadata ===")
            print(json.dumps(checkpoint["roboflow"], indent=2))

        # Check for training info
        if "epoch" in checkpoint:
            print(f"\nTrained for {checkpoint['epoch']} epochs")

        if "best_fitness" in checkpoint:
            print(f"Best fitness: {checkpoint['best_fitness']}")

    else:
        print("Checkpoint is not a dictionary. Type:", type(checkpoint))
        if hasattr(checkpoint, "__dict__"):
            print("Object attributes:", dir(checkpoint))

    return checkpoint


# Usage
model_path = "checkbox-nano.pt"  # Replace with your actual path
checkpoint = inspect_rfdetr_nano_model(model_path)


# Try to extract class information from different possible locations
def extract_classes_from_checkpoint(checkpoint):
    """
    Extract class information from various possible locations in the checkpoint
    """
    classes = None
    num_classes = None

    # Common locations for class information in Roboflow models
    if isinstance(checkpoint, dict):
        # Check for direct class names
        if "names" in checkpoint:
            classes = checkpoint["names"]
        elif "model_names" in checkpoint:
            classes = checkpoint["model_names"]
        elif "classes" in checkpoint:
            classes = checkpoint["classes"]

        # Check within model config
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            if "names" in checkpoint["config"]:
                classes = checkpoint["config"]["names"]
            if "num_classes" in checkpoint["config"]:
                num_classes = checkpoint["config"]["num_classes"]

        # Check within model metadata
        if "metadata" in checkpoint and isinstance(checkpoint["metadata"], dict):
            if "names" in checkpoint["metadata"]:
                classes = checkpoint["metadata"]["names"]

    return classes, num_classes


classes, num_classes = extract_classes_from_checkpoint(checkpoint)
print("\n=== Extracted Class Information ===")
print(f"Classes: {classes}")
print(
    f"Number of classes: {num_classes if num_classes else (len(classes) if classes else 'Unknown')}"
)
