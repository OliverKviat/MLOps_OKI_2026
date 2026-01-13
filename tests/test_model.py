import torch
import pytest

from adv_cookie_recipy.model import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [1, 32, 64, 128])
def test_model(batch_size: int) -> None:
    """Test that MyAwesomeModel produces correct output shape for different batch sizes."""
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {y.shape}"

def test_error_on_wrong_shape():
    """Test that model raises error on wrong input shape."""
    model = MyAwesomeModel()
    # Test with 2D tensor instead of 4D - should raise RuntimeError
    with pytest.raises(RuntimeError):
        x_wrong = torch.randn(1, 28)  # Wrong shape
        model(x_wrong)