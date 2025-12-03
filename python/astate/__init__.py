"""
Astate Client Python Interface

A Python client library for Astate tensor storage and retrieval system.
"""

from ._core import TensorStorage, TensorTableType, ShardedKey
from astate.table import TensorTable
from astate.parallel_config import Role, ParallelConfig
from astate import utils

__version__ = "0.1.0"

# Public API
__all__ = [
    # Core classes
    'TensorStorage',
    'TensorTableType',
    'ShardedKey',
    'TensorTable',

    # Parallel config classes
    'Role',
    'ParallelConfig',

    # Modules
    'utils',

    # Metadata
    '__version__',
    '__author__',
    '__email__'
]

# Convenience functions for common use cases
def create_table(name: str, parallel_config: ParallelConfig, table_type: TensorTableType = TensorTableType.IN_MEMORY) -> TensorTable:
    """
    Create a new table manager instance
    Args:
        name: Table name
        parallel_config: Parallel configuration for distributed operations
        table_type: Type of table implementation (IN_MEMORY, REMOTE)
    Returns:
        TensorTable: New table manager instance
    """
    return TensorTable(name, parallel_config, table_type)

def create_memory_table(name: str, parallel_config: ParallelConfig) -> TensorTable:
    """
    Create a new in-memory table manager instance
    Args:
        name: Table name
        parallel_config: Parallel configuration for distributed operations
    Returns:
        TensorTable: New in-memory table manager instance
    """
    return TensorTable(name, parallel_config, TensorTableType.IN_MEMORY)

def create_remote_table(name: str, parallel_config: ParallelConfig) -> TensorTable:
    """
    Create a new remote table manager instance

    Args:
        name: Table name
        parallel_config: Parallel configuration for distributed operations

    Returns:
        TensorTable: New remote table manager instance
    """
    return TensorTable(name, parallel_config, TensorTableType.REMOTE) 
