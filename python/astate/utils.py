"""
Utility functions for astate client.

This module provides various utility functions for tensor validation,
information extraction, and debugging support.
"""
import torch
from typing import List, Tuple, Union, Dict, Any, Optional
from astate._core import ShardedKey

def create_sharded_key(key: str, global_shape: List[int], global_offset: List[int]) -> ShardedKey:
    """
    Create a ShardedKey object.
    
    Args:
        key: String key
        global_shape: Global shape of the tensor
        global_offset: Global offset of the tensor
        
    Returns:
        ShardedKey: Created ShardedKey object
    """
    sharded_key = ShardedKey()
    sharded_key.key = key
    sharded_key.globalShape = global_shape
    sharded_key.globalOffset = global_offset
    return sharded_key

def validate_tensor(tensor: torch.Tensor) -> bool:
    """
    Validate if a tensor is valid for storage.
    
    Args:
        tensor: Tensor to validate
        
    Returns:
        bool: Whether the tensor is valid
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return False
    
    try:
        return tensor.numel() > 0
    except (RuntimeError, AttributeError):
        return False

def get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get comprehensive information about a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        dict: Dictionary containing tensor shape, dtype, device, etc.
        
    Raises:
        ValueError: If tensor is invalid
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
        
    try:
        info = {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'numel': tensor.numel(),
            'requires_grad': tensor.requires_grad,
            'is_contiguous': tensor.is_contiguous(),
            'element_size': tensor.element_size(),
        }
        
        # Safe addition of CUDA information
        if tensor.device.type == 'cuda':
            try:
                info['cuda_device'] = tensor.device.index
                info['is_cuda'] = True
            except Exception:
                info['is_cuda'] = True
        else:
            info['is_cuda'] = False
            
        return info
        
    except Exception as e:
        raise RuntimeError(f"Failed to get tensor info: {e}") from e

def validate_tensor_dict(tensor_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Validate a dictionary of tensors.
    
    Args:
        tensor_dict: Dictionary to validate
        
    Returns:
        bool: Whether all tensors are valid
    """
    if not isinstance(tensor_dict, dict):
        return False
        
    for key, tensor in tensor_dict.items():
        if not isinstance(key, str) or not validate_tensor(tensor):
            return False
    
    return True

def get_memory_usage(tensor: torch.Tensor) -> int:
    """
    Get memory usage of a tensor in bytes.
    
    Args:
        tensor: Input tensor
        
    Returns:
        int: Memory usage in bytes
        
    Raises:
        ValueError: If tensor is invalid
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    try:
        return tensor.numel() * tensor.element_size()
    except Exception as e:
        raise RuntimeError(f"Failed to calculate memory usage: {e}") from e

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                   rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Compare two tensors for equality within tolerance
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        bool: Whether tensors are equal within tolerance
        
    Raises:
        ValueError: If tensors are invalid
    """
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        raise ValueError("Both inputs must be torch.Tensor")
        
    if tensor1.shape != tensor2.shape:
        return False
    
    try:
        return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    except Exception as e:
        raise RuntimeError(f"Failed to compare tensors: {e}") from e

def summarize_tensors(tensors: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Generate summary statistics for a list of tensors.
    
    Args:
        tensors: List of tensors to summarize
        
    Returns:
        dict: Summary statistics
    """
    if not tensors:
        return {"count": 0, "total_memory": 0}
    
    valid_tensors = [t for t in tensors if validate_tensor(t)]
    
    total_memory = sum(get_memory_usage(t) for t in valid_tensors)
    shapes = [tuple(t.shape) for t in valid_tensors]
    dtypes = [str(t.dtype) for t in valid_tensors]
    devices = [str(t.device) for t in valid_tensors]
    
    return {
        "count": len(tensors),
        "valid_count": len(valid_tensors),
        "total_memory_bytes": total_memory,
        "total_memory_mb": total_memory / (1024 * 1024),
        "unique_shapes": list(set(shapes)),
        "unique_dtypes": list(set(dtypes)),
        "unique_devices": list(set(devices)),
    }
