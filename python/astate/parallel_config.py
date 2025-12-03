"""
Distributed training/inference configuration data structures

This module defines data structures for configuring distributed training and inference,
including size and rank information for various parallelism strategies.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class Role(Enum):
    """Execution role enumeration"""
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class ParallelConfig:
    """
    Distributed parallel configuration data structure
    
    Contains role size, role rank, and configuration information for various parallelism strategies.
    
    Attributes:
        role (Role): Execution role (training or inference)
        role_size (int): Total number of processes
        role_rank (int): Rank within the current execution role
        dp_size (int): Data parallel size
        dp_rank (int): Data parallel rank
        tp_size (int): Tensor parallel size
        tp_rank (int): Tensor parallel rank
        pp_size (int): Pipeline parallel size
        pp_rank (int): Pipeline parallel rank
        ep_size (int): Expert parallel size
        ep_rank (int): Expert parallel rank
        etp_size (int): Expert tensor parallel size
        etp_rank (int): Expert tensor parallel rank
    """
    role: Role
    role_size: int
    role_rank: int
    dp_size: int
    dp_rank: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    cp_size: int
    cp_rank: int
    ep_size: int
    ep_rank: int
    etp_size: int
    etp_rank: int

    def __post_init__(self):
        """Data validation"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the validity of configuration parameters"""
        # Validate that ranks cannot be negative
        ranks = [self.role_rank, self.dp_rank, 
                 self.tp_rank, self.pp_rank, self.ep_rank, self.cp_rank, self.etp_rank]
        if any(rank < 0 for rank in ranks):
            raise ValueError("All ranks must be non-negative")
        
        # Validate that sizes must be positive
        sizes = [self.role_size, self.dp_size, self.tp_size, 
                 self.pp_size, self.ep_size, self.cp_size, self.etp_size]
        if any(size <= 0 for size in sizes):
            raise ValueError("All size parameters must be positive")
        
        # Validate that ranks cannot exceed corresponding sizes
        if self.role_rank >= self.role_size:
            raise ValueError(f"Role rank {self.role_rank} must be less than role size {self.role_size}")
        
        if self.dp_rank >= self.dp_size:
            raise ValueError(f"Data parallel rank {self.dp_rank} must be less than data parallel size {self.dp_size}")
        
        if self.tp_rank >= self.tp_size:
            raise ValueError(f"Tensor parallel rank {self.tp_rank} must be less than tensor parallel size {self.tp_size}")
        
        if self.pp_rank >= self.pp_size:
            raise ValueError(f"Pipeline parallel rank {self.pp_rank} must be less than pipeline parallel size {self.pp_size}")
        
        if self.ep_rank >= self.ep_size:
            raise ValueError(f"Expert parallel rank {self.ep_rank} must be less than expert parallel size {self.ep_size}")
        
        if self.cp_rank >= self.cp_size:
            raise ValueError(f"Context parallel rank {self.cp_rank} must be less than context parallel size {self.cp_size}")
        
        if self.etp_rank >= self.etp_size:
            raise ValueError(f"Expert tensor parallel rank {self.etp_rank} must be less than expert tensor parallel size {self.etp_size}")
    
    @classmethod
    def create_training_config(
        cls,
        role_size: int,
        role_rank: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        cp_size: int = 1,
        cp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        etp_size: int = 1,
        etp_rank: int = 0
    ) -> 'ParallelConfig':
        """
        Create parallel configuration for training mode
        
        Args:
            role_size: Total number of processes
            role_rank: Role rank
            dp_size: Data parallel size, default is 1
            dp_rank: Data parallel rank, default is 0
            tp_size: Tensor parallel size, default is 1
            tp_rank: Tensor parallel rank, default is 0
            pp_size: Pipeline parallel size, default is 1
            pp_rank: Pipeline parallel rank, default is 0
            cp_size: Context parallel size, default is 1
            cp_rank: Context parallel rank, default is 0
            ep_size: Expert parallel size, default is 1
            ep_rank: Expert parallel rank, default is 0
            etp_size: Expert tensor parallel size, default is 1
            etp_rank: Expert tensor parallel rank, default is 0
            
        Returns:
            ParallelConfig: Parallel configuration instance for training mode
        """
        return cls(
            role=Role.TRAINING,
            role_size=role_size,
            role_rank=role_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            etp_size=etp_size,
            etp_rank=etp_rank
        )
    
    @classmethod
    def create_inference_config(
        cls,
        role_size: int,
        role_rank: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        tp_size: int = 1,
        tp_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        cp_size: int = 1,
        cp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        etp_size: int = 1,
        etp_rank: int = 0
    ) -> 'ParallelConfig':
        """
        Create parallel configuration for inference mode
        
        Args:
            role_size: Total number of processes
            role_rank: Role rank
            dp_size: Data parallel size, default is 1
            dp_rank: Data parallel rank, default is 0
            tp_size: Tensor parallel size, default is 1
            tp_rank: Tensor parallel rank, default is 0
            pp_size: Pipeline parallel size, default is 1
            pp_rank: Pipeline parallel rank, default is 0
            cp_size: Context parallel size, default is 1
            cp_rank: Context parallel rank, default is 0
            ep_size: Expert parallel size, default is 1
            ep_rank: Expert parallel rank, default is 0
            etp_size: Expert tensor parallel size, default is 1
            etp_rank: Expert tensor parallel rank, default is 0
            
        Returns:
            ParallelConfig: Parallel configuration instance for inference mode
        """
        return cls(
            role=Role.INFERENCE,
            role_size=role_size,
            role_rank=role_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            etp_size=etp_size,
            etp_rank=etp_rank
        )
    
    def is_training(self) -> bool:
        """Check if in training mode"""
        return self.role == Role.TRAINING
    
    def is_inference(self) -> bool:
        """Check if in inference mode"""
        return self.role == Role.INFERENCE
    
    def get_parallel_group_info(self) -> dict:
        """
        Get parallel group information
        
        Returns:
            dict: Dictionary containing all parallel group information
        """
        return {
            "role": self.role.value,
            "role_size": self.role_size,
            "role_rank": self.role_rank,
            "data_parallel": {"size": self.dp_size, "rank": self.dp_rank},
            "tensor_parallel": {"size": self.tp_size, "rank": self.tp_rank},
            "pipeline_parallel": {"size": self.pp_size, "rank": self.pp_rank},
            "context_parallel": {"size": self.cp_size, "rank": self.cp_rank},
            "expert_parallel": {"size": self.ep_size, "rank": self.ep_rank},
            "expert_tensor_parallel": {"size": self.etp_size, "rank": self.etp_rank}
        }
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"ParallelConfig(role={self.role.value}, "
            f"role_size={self.role_size}, "
            f"role_rank={self.role_rank}, "
            f"dp={self.dp_rank}/{self.dp_size}, "
            f"tp={self.tp_rank}/{self.tp_size}, "
            f"pp={self.pp_rank}/{self.pp_size}, "
            f"cp={self.cp_rank}/{self.cp_size}, "
            f"ep={self.ep_rank}/{self.ep_size}, "
            f"etp={self.etp_rank}/{self.etp_size})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return str(self) 