"""
Configuration converter module for converting between different config types.

This module provides utilities to convert between Python dataclass configs
and C++ struct configs for cross-language compatibility.
"""

from typing import Union
from astate._core import AParallelConfig, ARole
from astate.parallel_config import ParallelConfig, Role


def convert_parallel_config(config: ParallelConfig) -> AParallelConfig:
    """
    Convert Python ParallelConfig to C++ AParallelConfig.

    Args:
        config (ParallelConfig): Python parallel configuration object

    Returns:
        astate.AParallelConfig: C++ parallel configuration object

    Raises:
        ValueError: If the config contains invalid values
        TypeError: If the config is not a ParallelConfig instance
    """
    if not isinstance(config, ParallelConfig):
        raise TypeError(f"Expected ParallelConfig, got {type(config)}")

    # Convert Python Role enum to C++ ARole enum
    if config.role == Role.TRAINING:
        cpp_role = ARole.TRAINING
    elif config.role == Role.INFERENCE:
        cpp_role = ARole.INFERENCE
    else:
        raise ValueError(f"Unknown role: {config.role}")

    # Create AParallelConfig with all parameters
    cpp_config = AParallelConfig(
        role=cpp_role,
        role_size=config.role_size,
        role_rank=config.role_rank,
        dp_size=config.dp_size,
        dp_rank=config.dp_rank,
        tp_size=config.tp_size,
        tp_rank=config.tp_rank,
        pp_size=config.pp_size,
        pp_rank=config.pp_rank,
        cp_size=config.cp_size,
        cp_rank=config.cp_rank,
        ep_size=config.ep_size,
        ep_rank=config.ep_rank,
        etp_size=config.etp_size,
        etp_rank=config.etp_rank
    )

    return cpp_config


def convert_cpp_parallel_config(cpp_config: AParallelConfig) -> ParallelConfig:
    """
    Convert C++ AParallelConfig to Python ParallelConfig.

    Args:
        cpp_config (astate.AParallelConfig): C++ parallel configuration object

    Returns:
        ParallelConfig: Python parallel configuration object

    Raises:
        ValueError: If the config contains invalid values
        TypeError: If the config is not an AParallelConfig instance
    """
    if not isinstance(cpp_config, AParallelConfig):
        raise TypeError(f"Expected AParallelConfig, got {type(cpp_config)}")

    # Convert C++ ARole enum to Python Role enum
    if cpp_config.role == ARole.TRAINING:
        py_role = Role.TRAINING
    elif cpp_config.role == ARole.INFERENCE:
        py_role = Role.INFERENCE
    else:
        raise ValueError(f"Unknown C++ role: {cpp_config.role}")

    # Create ParallelConfig with all parameters
    py_config = ParallelConfig(
        role=py_role,
        role_size=cpp_config.role_size,
        role_rank=cpp_config.role_rank,
        dp_size=cpp_config.dp_size,
        dp_rank=cpp_config.dp_rank,
        tp_size=cpp_config.tp_size,
        tp_rank=cpp_config.tp_rank,
        pp_size=cpp_config.pp_size,
        pp_rank=cpp_config.pp_rank,
        cp_size=cpp_config.cp_size,
        cp_rank=cpp_config.cp_rank,
        ep_size=cpp_config.ep_size,
        ep_rank=cpp_config.ep_rank,
        etp_size=cpp_config.etp_size,
        etp_rank=cpp_config.etp_rank
    )

    return py_config


class ParallelConfigConverter:
    """
    Utility class for configuration conversions.

    Provides static methods for converting between different config types
    with validation and error handling.
    """

    @staticmethod
    def to_cpp_config(config: ParallelConfig) -> AParallelConfig:
        """
        Convert Python config to C++ config.

        Args:
            config: Python ParallelConfig instance

        Returns:
            C++ AParallelConfig instance
        """
        return convert_parallel_config(config)

    @staticmethod
    def to_python_config(cpp_config: AParallelConfig) -> ParallelConfig:
        """
        Convert C++ config to Python config.

        Args:
            cpp_config: C++ AParallelConfig instance

        Returns:
            Python ParallelConfig instance
        """
        return convert_cpp_parallel_config(cpp_config)

    @staticmethod
    def validate_compatibility(py_config: ParallelConfig, 
                             cpp_config: AParallelConfig) -> bool:
        """
        Validate that Python and C++ configs are equivalent.

        Args:
            py_config: Python ParallelConfig instance
            cpp_config: C++ AParallelConfig instance

        Returns:
            bool: True if configs are equivalent, False otherwise
        """
        try:
            # Convert Python to C++ and compare
            converted_cpp = convert_parallel_config(py_config)

            # Compare all fields
            return (
                converted_cpp.role == cpp_config.role and
                converted_cpp.role_size == cpp_config.role_size and
                converted_cpp.role_rank == cpp_config.role_rank and
                converted_cpp.dp_size == cpp_config.dp_size and
                converted_cpp.dp_rank == cpp_config.dp_rank and
                converted_cpp.tp_size == cpp_config.tp_size and
                converted_cpp.tp_rank == cpp_config.tp_rank and
                converted_cpp.pp_size == cpp_config.pp_size and
                converted_cpp.pp_rank == cpp_config.pp_rank and
                converted_cpp.cp_size == cpp_config.cp_size and
                converted_cpp.cp_rank == cpp_config.cp_rank and
                converted_cpp.ep_size == cpp_config.ep_size and
                converted_cpp.ep_rank == cpp_config.ep_rank and
                converted_cpp.etp_size == cpp_config.etp_size and
                converted_cpp.etp_rank == cpp_config.etp_rank
            )
        except (ValueError, TypeError):
            return False


# Convenience functions for direct usage
def py_to_cpp(config: ParallelConfig) -> AParallelConfig:
    """Shorthand for converting Python config to C++ config."""
    return convert_parallel_config(config)


def cpp_to_py(cpp_config: AParallelConfig) -> ParallelConfig:
    """Shorthand for converting C++ config to Python config."""
    return convert_cpp_parallel_config(cpp_config) 