import torch
import torch.nn as nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    """A simple CNN model for MNIST digit classification."""
    
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28x28 -> 14x14 -> 7x7 after two pooling layers
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits (0-9)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        """Forward pass through the network."""
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64*7*7)
        
        # First fully connected layer with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer (no activation, will be handled by loss function)
        x = self.fc2(x)
        
        return x