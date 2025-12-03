"""
Astate Client Core Module - C++ bindings
"""
try:
    from .astate_cpp import TensorTable, TensorTableType, ShardedKey, TorchTensorMeta, TensorStorage, AParallelConfig, ARole
    __all__ = ['TensorTable', 'TensorTableType', 'ShardedKey', 'TorchTensorMeta', 'TensorStorage', 'AParallelConfig', 'ARole']
except ImportError as e:
    import sys
    print(f"Error: Failed to import C++ extension module.\n"
          f"Original error: {str(e)}\n"
          f"This usually means the module was not built or installed correctly.\n"
          f"Please make sure you have built the C++ extension module and it is placed in the correct location.",
          file=sys.stderr)
    raise
