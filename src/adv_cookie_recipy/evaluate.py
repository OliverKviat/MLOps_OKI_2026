#!/usr/bin/env python3
"""
Script to evaluate a trained model on test data.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model_path: Path, test_images_path: Path, test_targets_path: Path):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the trained model file
        test_images_path: Path to test images tensor
        test_targets_path: Path to test targets tensor
    """
    try:
        # Check if files exist
        for path in [model_path, test_images_path, test_targets_path]:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load the model
        # Note: This assumes the model was saved with torch.save()
        # You may need to adjust this based on your model architecture
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        logger.info(f"Loading test data from: {test_images_path} and {test_targets_path}")
        
        # Load test data
        test_images = torch.load(test_images_path, map_location='cpu')
        test_targets = torch.load(test_targets_path, map_location='cpu')
        
        logger.info(f"Test images shape: {test_images.shape}")
        logger.info(f"Test targets shape: {test_targets.shape}")
        
        # Run inference
        with torch.no_grad():
            predictions = model(test_images)
            
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # Calculate metrics (basic example - adjust based on your task)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Classification task
            predicted_classes = torch.argmax(predictions, dim=1)
            accuracy = (predicted_classes == test_targets).float().mean().item()
            logger.info(f"Accuracy: {accuracy:.4f}")
        else:
            # Regression task
            mse = torch.nn.functional.mse_loss(predictions.squeeze(), test_targets.float()).item()
            logger.info(f"MSE: {mse:.4f}")
            
        logger.info("Evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return False


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test data")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("test_images_path", type=str, default="/test_images.pt", 
                       help="Path to test images file")
    parser.add_argument("test_targets_path", type=str, default="/test_targets.pt",
                       help="Path to test targets file")
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    model_path = Path(args.model_path)
    test_images_path = Path(args.test_images_path)
    test_targets_path = Path(args.test_targets_path)
    
    logger.info("Starting model evaluation...")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test images path: {test_images_path}")
    logger.info(f"Test targets path: {test_targets_path}")
    
    success = evaluate_model(model_path, test_images_path, test_targets_path)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
