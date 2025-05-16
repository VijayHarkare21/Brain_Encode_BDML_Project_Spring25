import torch
import numpy as np

def create_random_tensor(shape, min_val=-30.0, max_val=30.0, 
                         dtype=torch.float32):
    """
    Creates a Torch tensor of the given shape with random numbers
    between min_val and max_val (uniform).

    Parameters:
        shape (tuple of ints): The desired shape.
        min_val (float): Lower bound (inclusive).
        max_val (float): Upper bound (exclusive).
        dtype (torch.dtype): The desired data type.

    Returns:
        torch.Tensor: A tensor with values in [min_val, max_val).
    """
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # Option A: scale torch.rand
    return (max_val - min_val) * torch.rand(*shape, dtype=dtype, device=device) + min_val
    
def create_random_array(shape, min_val=-30.0, max_val=30.0):
    """
    Creates a NumPy array of the given shape with random numbers
    between min_val and max_val.

    Parameters:
        shape (tuple): The shape of the array.
        min_val (int or float): The minimum value of the random numbers.
        max_val (int or float): The maximum value of the random numbers (exclusive).

    Returns:
        numpy.ndarray: An array with random numbers.
    """

    return (max_val - min_val) * np.random.rand(*shape) + min_val