import torch
import pytest
import os.path
from torch.utils.data import Dataset
from adv_cookie_recipy.data import MyDataset, corrupt_mnist


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset), "MyDataset should be an instance of torch.utils.data.Dataset"

def test_data():
    """Test the corrupt_mnist function."""
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Expected 30000 training samples"
    assert len(test) == 5000, "Expected 5000 test samples"
    
    # Test a few samples from each dataset to avoid long test times
    train_sample_x, train_sample_y = train[0]
    test_sample_x, test_sample_y = test[0]
    
    assert train_sample_x.shape == (1, 28, 28), "Expected training input shape (1, 28, 28)"
    assert train_sample_y in range(10), "Expected training target in range 0-9"
    assert test_sample_x.shape == (1, 28, 28), "Expected test input shape (1, 28, 28)"
    assert test_sample_y in range(10), "Expected test target in range 0-9"
    
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "Expected training targets to cover all classes 0-9"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "Expected test targets to cover all classes 0-9"