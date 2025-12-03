"""
Basic usage example of Astate client with new factory-based interface.

This example demonstrates how to use the Astate client with the new
TensorTableFactory interface and improved error handling.
"""
import torch
from astate import TensorTable, create_table, create_memory_table, create_remote_table
from astate.utils import get_tensor_info, create_sharded_key
from astate.table import TensorTableType
from astate.parallel_config import ParallelConfig

def demonstrate_basic_operations():
    """Demonstrate basic tensor operations."""
    print("🚀 Basic Tensor Operations Demo")
    print("=" * 40)
    
    # Create table using different methods
    print("\n📋 Creating tables...")

    parallel_config = ParallelConfig.create_training_config(role_size=1, role_rank=0)
    #
    # Method 1: Using TableManager constructor
    table1 = create_table("example_table_1", parallel_config, TensorTableType.IN_MEMORY)
    print(f"✅ Created table1: {table1}")
    
    # Method 2: Using class methods
    table2 = create_memory_table("example_table_2", parallel_config)
    print(f"✅ Created table2: {table2}")
    
    # Create test tensors
    print("\n📦 Creating test tensors...")
    tensor1 = torch.randn(3, 4, dtype=torch.float32)
    tensor2 = torch.ones(4, 4, dtype=torch.float32)
    
    print(f"tensor1 info: {get_tensor_info(tensor1)}")
    print(f"tensor2 info: {get_tensor_info(tensor2)}")
    
    # Single tensor operations
    print("\n💾 Single tensor operations...")
    
    # Store tensors
    sharded_key1 = create_sharded_key("feature1", [3, 4], [0, 0])
    success1 = table1.put(seq_id=1, key=sharded_key1, tensor=tensor1)
    sharded_key2 = create_sharded_key("feature2", [4, 4], [0, 0])
    success2 = table1.put(seq_id=1, key=sharded_key2, tensor=tensor2)
    print(f"Store tensor1: {'✅ success' if success1 else '❌ failed'}")
    print(f"Store tensor2: {'✅ success' if success2 else '❌ failed'}")
    
    # Retrieve tensors
    retrieved1 = table1.get(seq_id=1, key=sharded_key1, tensor=torch.randn(3, 4, dtype=torch.float32))
    retrieved2 = table1.get(seq_id=1, key=sharded_key2, tensor=torch.randn(4, 4, dtype=torch.float32))
    
    if retrieved1 is not None:
        print("✅ Retrieved tensor1 successfully")
        print(f"  Original: {tensor1.shape} {tensor1.dtype}")
        print(f"  Retrieved: {retrieved1.shape} {retrieved1.dtype}")
        is_equal = torch.allclose(tensor1, retrieved1)
        print(f"  Data consistency: {'✅ passed' if is_equal else '❌ failed'}")
    
    if retrieved2 is not None:
        print("✅ Retrieved tensor2 successfully")
        print(f"  Original: {tensor2.shape} {tensor2.dtype}")
        print(f"  Retrieved: {retrieved2.shape} {retrieved2.dtype}")
        is_equal = torch.equal(tensor2, retrieved2)
        print(f"  Data consistency: {'✅ passed' if is_equal else '❌ failed'}")

def demonstrate_batch_operations():
    """Demonstrate batch tensor operations."""
    print("\n📦 Batch Operations Demo")
    print("=" * 40)
    
    table = create_table("batch_table", TensorTableType.IN_MEMORY)
    
    # Create multiple tensors
    tensor_dict = {
        "weights": torch.randn(64, 32, dtype=torch.float32),
        "biases": torch.randn(32, dtype=torch.float32),
        "indices": torch.randn(10, dtype=torch.float32),
        "mask": torch.randn(10, 5, dtype=torch.float32),
    }
    
    print(f"📋 Created tensor dictionary:")
    print("Input tensors", tensor_dict)
    
    # Batch store
    print("\n💾 Batch storage...")
    try:
        tensor_pairs = [(key, tensor_dict[key].clone()) for key in tensor_dict.keys()]
        success = table.multi_put(seq_id=2, tensor_pairs=tensor_pairs)
        print(f"Batch store: {'✅ success' if success else '❌ failed'}")
    except Exception as e:
        print(f"❌ Batch store failed: {e}")
        return
    
    # Batch retrieve
    print("\n📤 Batch retrieval...")
    try:
        tensor_pairs = [(key, torch.randn(tensor_dict[key].shape, dtype=tensor_dict[key].dtype)) for key in tensor_dict.keys()]
        results = table.multi_get(seq_id=2, tensor_pairs=tensor_pairs)
        print(f"✅ Retrieved {len(results)} tensors")
        
        print(debug_tensor_dict(results, "Retrieved tensors"))
        
        # Verify data consistency
        print("\n🔍 Data consistency check...")
        all_consistent = True
        for key in keys:
            if key in results:
                original = tensor_dict[key]
                retrieved = results[key]
                
                if original.dtype == torch.bool or original.dtype.is_integer:
                    is_equal = torch.equal(original, retrieved)
                else:
                    is_equal = torch.allclose(original, retrieved)
                
                print(f"  {key}: {'✅ consistent' if is_equal else '❌ inconsistent'}")
                all_consistent = all_consistent and is_equal
            else:
                print(f"  {key}: ❌ not found")
                all_consistent = False
        
        print(f"Overall consistency: {'✅ passed' if all_consistent else '❌ failed'}")
        
    except Exception as e:
        print(f"❌ Batch retrieve failed: {e}")

def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n⚠️  Error Handling Demo")
    print("=" * 40)
    
    table = create_table("error_test_table", TensorTableType.IN_MEMORY)
    
    # Test invalid tensor
    print("\n🧪 Testing invalid tensor handling...")
    try:
        table.put(1, "invalid", "not_a_tensor")  # This should fail
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")
    
    # Test empty tensor
    print("\n🧪 Testing empty tensor handling...")
    try:
        empty_tensor = torch.empty(0)
        table.put(1, "empty", empty_tensor)  # This should fail
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")
    
    # Test non-existent key
    print("\n🧪 Testing non-existent key...")
    sharded_key = create_sharded_key("non_existent_key", [3, 3], [0, 0])
    result = table.get(1, sharded_key, torch.randn(3, 3, dtype=torch.float32))
    print(f"Non-existent key result: {result}")

def demonstrate_table_types():
    """Demonstrate different table types."""
    print("\n🏗️  Table Types Demo")
    print("=" * 40)
    
    # In-memory table
    print("\n📝 In-memory table...")
    memory_table = create_table("memory_table", TensorTableType.IN_MEMORY)
    print(f"✅ Created: {memory_table}")
    
    # Test storing and retrieving
    test_tensor = torch.randn(3, 3)
    sharded_key = create_sharded_key("test", [3, 3], [0, 0])
    memory_table.put(1, sharded_key, test_tensor)
    retrieved = memory_table.get(1, sharded_key, torch.randn(3, 3, dtype=torch.float32))
    print(f"Memory table works: {retrieved is not None}")
    
    # Remote table (will fail for now)
    print("\n🌐 Remote table...")
    try:
        parallel_config = ParallelConfig.create_training_config(role_size=1, role_rank=0)
        remote_table = create_remote_table("remote_table", parallel_config)
        print(f"✅ Created: {remote_table}")
    except Exception as e:
        print(f"❌ Remote table creation failed (expected): {e}")

def main():
    """Main demonstration function."""
    print("🎯 Astate Client Advanced Demo")
    print("=" * 50)
    
    try:
        demonstrate_basic_operations()
        demonstrate_batch_operations()
        demonstrate_error_handling()
        demonstrate_table_types()
        
        print("\n" + "=" * 50)
        print("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 