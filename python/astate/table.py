"""
High-level Python API for TensorTable operations.

This module provides a Python interface for managing TensorTable instances
and performing tensor storage/retrieval operations using the new factory-based design.
"""
import torch
from typing import Dict, List, Union, Optional, Tuple, Generator
from astate._core import TensorStorage, TensorTableType, ShardedKey, TorchTensorMeta, TensorTable as CoreTensorTable
from astate.parallel_config import ParallelConfig
from astate.config_converter import convert_parallel_config

class TensorTable:
    """
    High-level Python API for managing TensorTable instances.

    This class provides a simplified interface for tensor storage and retrieval
    operations using the underlying C++ TensorTable implementation.

    Attributes:
        name (str): The name of the managed table
        table_type (Union[str, TensorTableType]): The type of table implementation
        parallel_config (ParallelConfig): Parallel configuration for distributed operations

    Example:
        >>> from astate.parallel_config import ParallelConfig
        >>> config = ParallelConfig.create_training_config(world_size=4, global_rank=0, role_rank=0)
        >>> table = TensorTable("my_table", parallel_config=config)
        >>> tensor = torch.randn(3, 4)
        >>> success = table.put(1, "key1", tensor)
        >>> retrieved = table.get(1, "key1")
    """
    _table: CoreTensorTable
    _name: str
    _type: TensorTableType
    _type_str: str
    _parallel_config: ParallelConfig

    def __init__(self, 
                 name: str, 
                 parallel_config: ParallelConfig,
                 table_type: Union[str, TensorTableType] = TensorTableType.IN_MEMORY):
        """
        Create or get a TensorTable instance.

        Args:
            name: Table name
            parallel_config: Parallel configuration for distributed operations (required)
            table_type: Type of table implementation (TensorTableType enum or string)

        Raises:
            ValueError: If failed to create or get table or unsupported table type
            TypeError: If parallel_config is not a ParallelConfig instance
        """
        # Handle both string and enum inputs
        if isinstance(table_type, str):
            # Map string types to enum values for backward compatibility
            type_mapping = {
                "in_memory": TensorTableType.IN_MEMORY,
                "memory": TensorTableType.IN_MEMORY,  # alias
                "remote": TensorTableType.REMOTE,
            }

            if table_type not in type_mapping:
                raise ValueError(f"Unsupported table type: {table_type}. "
                               f"Supported types: {list(type_mapping.keys())}")

            tensor_table_type = type_mapping[table_type]
            self._type_str = table_type
        else:
            # Direct enum input
            tensor_table_type = table_type
            if tensor_table_type == TensorTableType.IN_MEMORY:
                self._type_str = "in_memory"
            elif tensor_table_type == TensorTableType.REMOTE:
                self._type_str = "remote"
            else:
                raise ValueError(f"Unsupported TensorTableType: {tensor_table_type}")

        # Validate parallel_config parameter
        if not isinstance(parallel_config, ParallelConfig):
            raise TypeError(f"parallel_config must be a ParallelConfig instance, got {type(parallel_config)}")

        # Convert Python ParallelConfig to C++ AParallelConfig
        try:
            cpp_parallel_config = convert_parallel_config(parallel_config)
        except Exception as e:
            raise ValueError(f"Failed to convert parallel config: {e}")

        try:
            self._table = TensorStorage.create_tensor_storage(cpp_parallel_config).register_table(tensor_table_type, name)
        except Exception as e:
            raise ValueError(f"Failed to create or get table '{name}': {e}")

        if self._table is None:
            raise ValueError(f"Failed to create or get table: {name}")

        self._name = name
        self._type = tensor_table_type
        self._parallel_config = parallel_config

    def put(self,
            seq_id: int,
            key: ShardedKey,
            tensor: torch.Tensor) -> bool:
        """
        Store a single tensor.

        Args:
            seq_id: Sequence ID
            key: ShardedKey
            tensor: Tensor to store

        Returns:
            bool: Whether successful

        Raises:
            ValueError: If tensor is invalid
            RuntimeError: If storage operation fails
        """
        if tensor is None or not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        # Validate tensor
        try:
            if tensor.numel() == 0:
                raise ValueError("Cannot store empty tensor")
        except (RuntimeError, AttributeError):
            raise ValueError("Invalid tensor: cannot determine size")

        try:
            return self._table.put(seq_id, key, tensor)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid input for put operation: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to put tensor: {e}") from e

    def get(self,
            seq_id: int,
            key: ShardedKey,
            tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get a single tensor.

        Args:
            seq_id: Sequence ID
            key: ShardedKey
            tensor: Tensor to store the retrieved tensor

        Returns:
            torch.Tensor: Retrieved tensor, None if not exists

        Note:
            For tensors with existing storage, in-place updates should ideally be performed. 

        Raises:
            RuntimeError: If retrieval operation fails
        """
        try:
            is_success = self._table.get(seq_id, key, tensor)

            # Safe check for tensor validity
            if not is_success:
                return None
            try:
                return tensor
            except (RuntimeError, AttributeError):
                # Handle case where tensor is invalid or corrupted
                return None
        except Exception as e:
            raise RuntimeError(f"Failed to get tensor: {e}") from e

    def get_tensor(self,
                   seq_id: int,
                   key: ShardedKey,
                   dtype: torch.dtype,
                   size: Tuple[int, ...],
                   device: torch.device) -> Optional[torch.Tensor]:
        """
        Get a single tensor for cases where no pre-allocated tensor with storage is available.

        Args:
            seq_id: Sequence ID
            key: ShardedKey for the tensor
            dtype: Data type of the tensor to retrieve
            size: Shape/size of the tensor to retrieve
            device: Target device for the returned tensor

        Returns:
            torch.Tensor: Retrieved tensor, None if not exists

        Raises:
            RuntimeError: If retrieval operation fails
        """
        try:
            # Create TorchTensorMeta with the provided metadata
            tensor_meta = TorchTensorMeta()
            tensor_meta.dtype = dtype
            tensor_meta.size = list(size)  # Convert tuple to list for std::vector
            tensor_meta.device = device

            # Call the underlying C++ get_tensor method
            return self._table.get_tensor(seq_id, key, tensor_meta)
        except Exception as e:
            raise RuntimeError(f"Failed to get tensor: {e}") from e

    def multi_put(self,
                 seq_id: int,
                 tensor_pairs: List[Tuple[ShardedKey, torch.Tensor]]) -> bool:
        """
        Store multiple tensors in batch.

        Warning:
            This function may have stability issues with memory management.
            For critical applications, consider using individual put() calls.

        Args:
            seq_id: Sequence ID
            tensor_pairs: List of key-tensor pairs

        Returns:
            bool: Whether all successful

        Raises:
            ValueError: If input is invalid
            RuntimeError: If storage operation fails
        """
        if not tensor_pairs:
            return True

        try:
            for key, tensor in tensor_pairs:
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Value for key '{key.key}' must be a torch.Tensor")

                # Additional validation
                try:
                    if tensor.numel() == 0:
                        raise ValueError(f"Empty tensor for key '{key.key}'")
                except (RuntimeError, AttributeError):
                    raise ValueError(f"Invalid tensor for key '{key.key}': cannot determine size")

            return self._table.multi_put(seq_id, tensor_pairs)

        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid input for multi_put: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to multi_put tensors: {e}") from e

    def multi_get(self,
                 seq_id: int,
                 tensor_pairs: List[Tuple[ShardedKey, torch.Tensor]]) -> List[Tuple[ShardedKey, torch.Tensor]]:
        """
        Get multiple tensors in batch.

        Warning:
            This function may have stability issues with memory management.
            For critical applications, consider using individual get() calls.
        
        Args:
            seq_id: Sequence ID
            tensor_pairs: List of key-tensor pairs to retrieve
            
        Returns:
            List[Tuple[ShardedKey, torch.Tensor]]: List of key-tensor pairs
            
        Note:
            For tensors with existing storage, in-place updates should ideally be performed. 
            
        Raises:
            RuntimeError: If retrieval operation fails
        """
        if not tensor_pairs:
            return []
            
        try:
            for key, tensor in tensor_pairs:
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Value for key '{key.key}' must be a torch.Tensor")
                
                # Additional validation
                try:
                    if tensor.numel() == 0:
                        raise ValueError(f"Empty tensor for key '{key.key}'")
                except (RuntimeError, AttributeError):
                    raise ValueError(f"Invalid tensor for key '{key.key}': cannot determine size")

            self._table.multi_get(seq_id, tensor_pairs)
            return tensor_pairs
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid input for multi_get: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to multi_get tensors: {e}") from e
    
    def multi_get_tensors(self, 
                   seq_id: int, 
                   tensor_pairs: List[Tuple[ShardedKey, torch.dtype, Tuple[int, ...], torch.device]]) -> List[Tuple[ShardedKey, torch.Tensor]]:
        """
        Get multiple tensors in batch for cases where no pre-allocated tensors with storage are available.
        
        Args:
            seq_id: Sequence ID
            tensor_pairs: List of tuples, each containing:
                - ShardedKey: Key for the tensor
                - torch.dtype: Data type of the tensor to retrieve
                - Tuple[int, ...]: Shape/size of the tensor to retrieve  
                - torch.device: Target device for the returned tensor
                
        Returns:
            List[Tuple[ShardedKey, torch.Tensor]]: List of key-tensor pairs
            
        Raises:
            RuntimeError: If retrieval operation fails
        """
        if not tensor_pairs or len(tensor_pairs) == 0:
            return []
            
        try:
            # Validate input parameters and create tensor meta list
            tensor_meta_list = []
            for key, dtype, size, device in tensor_pairs:
                # Create TorchTensorMeta for this tensor
                tensor_meta = TorchTensorMeta()
                tensor_meta.dtype = dtype
                tensor_meta.size = list(size)  # Convert tuple to list for std::vector
                tensor_meta.device = device
                
                tensor_meta_list.append((key, tensor_meta))
            
            # Call the underlying C++ multi_get_tensor method
            return self._table.multi_get_tensor(seq_id, tensor_meta_list)
        except Exception as e:
            raise RuntimeError(f"Failed to multi_get_tensors: {e}") from e

    def complete(self, seq_id: int) -> None:
        """
        Complete operations for a given sequence ID.
        
        Warning:
            This interface "may" be a blocking interface, depending on the underlying implementation mode:
            - In some modes, this interface needs to wait until both read and write groups complete
            - In some modes, this interface needs to wait until data in memory is persisted to storage
            - The actual behavior depends on the table type and configuration
        
        Args:
            seq_id: Sequence ID to complete operations for
            
        Raises:
            RuntimeError: If completion operation fails
        """
        try:
            self._table.complete(seq_id)
        except Exception as e:
            raise RuntimeError(f"Failed to complete operations for seq_id {seq_id}: {e}") from e

    def scan_tensor_meta(self, seq_id: int) -> Generator[Tuple[str, torch.dtype, torch.Size], None, None]:
        """
        Scan all tensor metadata under a given sequence ID.
        
        This method returns a generator that yields TensorMeta tuples containing
        metadata information for all tensors stored under the specified sequence ID.
        
        Args:
            seq_id: Sequence ID to scan tensor metadata for
            
        Yields:
            Tuple[str, torch.dtype, torch.Size]: TensorMeta tuple containing:
                - key (str): Tensor key as string
                - dtype (torch.dtype): Data type of the tensor
                - size (torch.Size): Shape/size of the tensor
                
        Raises:
            RuntimeError: If scan operation fails
        """
        try:
            # Call the underlying C++ scan_tensor_meta method
            # Returns List[Tuple[str, TorchTensorMeta]]
            result_list = self._table.scan_tensor_meta(seq_id)
            
            # Convert results to generator format
            for key, tensor_meta in result_list:
                # Extract metadata from TorchTensorMeta object
                dtype = tensor_meta.dtype
                size = torch.Size(tensor_meta.size)  # Convert std::vector<int64_t> to torch.Size
                
                yield (key, dtype, size)
                
        except Exception as e:
            raise RuntimeError(f"Failed to scan tensor metadata for seq_id {seq_id}: {e}") from e
    
    @property
    def name(self) -> str:
        """Get table name."""
        return self._name
    
    @property
    def table_type(self) -> str:
        """Get table type as string."""
        return self._type_str
    
    @property
    def table_type_enum(self) -> TensorTableType:
        """Get table type as enum."""
        return self._type
    
    @property
    def parallel_config(self) -> ParallelConfig:
        """Get current parallel configuration."""
        return self._parallel_config
    
    def update_parallel_config(self, new_config: ParallelConfig) -> None:
        """
        Update the parallel configuration for the table.
        
        Args:
            new_config: New parallel configuration to apply
            
        Raises:
            ValueError: If the new configuration is invalid
        """
        if not isinstance(new_config, ParallelConfig):
            raise ValueError("new_config must be a ParallelConfig instance")
        
        # Store the new configuration
        self._parallel_config = new_config
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TableManager(name='{self._name}', type='{self._type_str}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"astate TableManager '{self._name}' ({self._type_str})"
