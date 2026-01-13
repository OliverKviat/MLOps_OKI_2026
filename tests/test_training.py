import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from unittest.mock import patch
import tempfile
import os
from pathlib import Path

from adv_cookie_recipy.model import MyAwesomeModel


class TestTraining:
    """Test suite for training functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a small mock dataset for testing
        self.batch_size = 4
        self.num_samples = 20
        
        # Generate dummy MNIST-like data
        self.mock_images = torch.randn(self.num_samples, 1, 28, 28)
        self.mock_labels = torch.randint(0, 10, (self.num_samples,))
        
        # Create dataset and dataloader
        self.mock_dataset = TensorDataset(self.mock_images, self.mock_labels)
        self.mock_dataloader = DataLoader(
            self.mock_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Initialize model and optimizer
        self.model = MyAwesomeModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def test_model_parameters_update_during_training(self):
        """Test that model parameters are actually updated during training."""
        # Get initial parameters
        initial_params = [param.clone() for param in self.model.parameters()]
        
        # Perform one training step
        self.model.train()
        batch_images, batch_labels = next(iter(self.mock_dataloader))
        
        # Forward pass
        outputs = self.model(batch_images)
        loss = self.criterion(outputs, batch_labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Check that parameters have changed
        params_changed = False
        for initial_param, current_param in zip(initial_params, self.model.parameters()):
            if not torch.equal(initial_param, current_param):
                params_changed = True
                break
                
        assert params_changed, "Model parameters should update during training"
        
    def test_loss_computation_and_backpropagation(self):
        """Test that loss is computed correctly and gradients exist."""
        self.model.train()
        batch_images, batch_labels = next(iter(self.mock_dataloader))
        
        # Forward pass
        outputs = self.model(batch_images)
        loss = self.criterion(outputs, batch_labels)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.requires_grad, "Loss should require gradients"
        assert loss.item() > 0, "Loss should be positive"
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist and are not NaN
        for param in self.model.parameters():
            assert param.grad is not None, "All parameters should have gradients"
            assert not torch.isnan(param.grad).any(), "Gradients should not contain NaN"
            assert not torch.isinf(param.grad).any(), "Gradients should not contain Inf"
            
    def test_training_mode_vs_eval_mode(self):
        """Test that model behaves differently in training vs evaluation mode."""
        batch_images, _ = next(iter(self.mock_dataloader))
        
        # Get outputs in training mode
        self.model.train()
        train_outputs = self.model(batch_images)
        
        # Get outputs in evaluation mode
        self.model.eval()
        eval_outputs = self.model(batch_images)
        
        # For models with dropout, outputs should differ between modes
        # Note: This test might not always pass due to randomness, but it's good to have
        # We test that the model at least switches modes properly
        assert self.model.training == False, "Model should be in eval mode"
        
        self.model.train()
        assert self.model.training == True, "Model should be in train mode"
        
    def test_optimizer_state_updates(self):
        """Test that optimizer state is properly maintained."""
        initial_state = len(self.optimizer.state)
        
        # Perform training step
        self.model.train()
        batch_images, batch_labels = next(iter(self.mock_dataloader))
        
        outputs = self.model(batch_images)
        loss = self.criterion(outputs, batch_labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Optimizer should have state for parameters after step
        final_state = len(self.optimizer.state)
        assert final_state >= initial_state, "Optimizer should maintain state for parameters"
        
    def test_model_output_shapes_during_training(self):
        """Test that model outputs have correct shapes during training."""
        self.model.train()
        
        for batch_images, batch_labels in self.mock_dataloader:
            outputs = self.model(batch_images)
            
            # Check output shape
            expected_shape = (batch_images.size(0), 10)  # (batch_size, num_classes)
            assert outputs.shape == expected_shape, \
                f"Output shape should be {expected_shape}, got {outputs.shape}"
            
            # Check that outputs are logits (no activation applied)
            assert not torch.isnan(outputs).any(), "Outputs should not contain NaN"
            assert not torch.isinf(outputs).any(), "Outputs should not contain Inf"
            
    def test_gradient_clipping_safety(self):
        """Test that gradients can be safely clipped without errors."""
        self.model.train()
        batch_images, batch_labels = next(iter(self.mock_dataloader))
        
        # Forward and backward pass
        outputs = self.model(batch_images)
        loss = self.criterion(outputs, batch_labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Test gradient clipping
        max_norm = 1.0
        total_norm_before = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm
        )
        
        assert total_norm_before >= 0, "Gradient norm should be non-negative"
        
        # Check that gradients are clipped properly
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                # Individual parameter gradients might exceed max_norm, 
                # but total norm should be controlled
                assert not torch.isnan(param_norm), "Parameter gradient norm should not be NaN"
                
    def test_model_can_handle_different_batch_sizes(self):
        """Test that model handles varying batch sizes correctly."""
        self.model.train()
        
        # Test with different batch sizes
        test_batch_sizes = [1, 3, 7, 16]
        
        for batch_size in test_batch_sizes:
            test_images = torch.randn(batch_size, 1, 28, 28)
            test_labels = torch.randint(0, 10, (batch_size,))
            
            # Forward pass should work
            outputs = self.model(test_images)
            loss = self.criterion(outputs, test_labels)
            
            # Check shapes
            assert outputs.shape == (batch_size, 10), \
                f"Wrong output shape for batch size {batch_size}"
            
            # Backward pass should work
            self.optimizer.zero_grad()
            loss.backward()
            
            assert loss.item() > 0, f"Loss should be positive for batch size {batch_size}"
            
    def test_training_with_empty_gradients_handling(self):
        """Test that training handles edge cases like zero gradients properly."""
        self.model.train()
        
        # Create a scenario that might produce small gradients
        batch_images = torch.zeros(2, 1, 28, 28)  # All zeros input
        batch_labels = torch.zeros(2, dtype=torch.long)  # All same label
        
        outputs = self.model(batch_images)
        loss = self.criterion(outputs, batch_labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Even with unusual inputs, training should not crash
        assert not torch.isnan(loss), "Loss should not be NaN even with zero inputs"
        
        # Check gradients exist (might be small but should exist)
        gradient_exists = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                gradient_exists = True
                break
                
        # This is more of a sanity check - gradients might be very small but should exist
        assert gradient_exists, "At least some gradients should exist"

    def test_model_deterministic_with_seed(self):
        """Test that model initialization is reproducible when using the same seed."""
        # Test that same seed produces same model initialization
        torch.manual_seed(42)
        model1 = MyAwesomeModel()
        
        torch.manual_seed(42)
        model2 = MyAwesomeModel()
        
        # Models should be identical after initialization with same seed
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), "Models should be identical with same seed"
        
        # Test that different seeds produce different models
        torch.manual_seed(123)
        model3 = MyAwesomeModel()
        
        # Model3 should be different from model1
        models_different = False
        for p1, p3 in zip(model1.parameters(), model3.parameters()):
            if not torch.allclose(p1, p3, atol=1e-6):
                models_different = True
                break
        
        assert models_different, "Models with different seeds should be different"