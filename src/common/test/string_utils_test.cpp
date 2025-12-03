#include "common/string_utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace astate {
class StringUtilsTest : public ::testing::Test {
 protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(StringUtilsTest, PointerToHexString) {
    void* null_ptr = nullptr;
    std::string null_result = PointerToHexString(null_ptr);
    EXPECT_EQ(null_result, "0x0000000000000000") << "Null pointer should be converted to 0x0000000000000000";

    int test_value = 42;
    void* test_ptr = &test_value;
    std::string test_result = PointerToHexString(test_ptr);

    EXPECT_EQ(test_result.substr(0, 2), "0x") << "Result should start with '0x'";
    EXPECT_EQ(test_result.length(), 18) << "Result should be 18 characters long (0x + 16 hex digits)";

    for (size_t i = 2; i < test_result.length(); ++i) {
        char c = test_result[i];
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))
            << "Character at position " << i << " should be a valid hex digit";
    }

    char small_value = 'A';
    void* small_ptr = &small_value;
    std::string small_result = PointerToHexString(small_ptr);
    EXPECT_EQ(small_result.substr(0, 2), "0x") << "Small pointer should start with '0x'";
    EXPECT_EQ(small_result.length(), 18) << "Small pointer result should be 18 characters long";

    // 测试大数值指针（模拟64位地址）
    uintptr_t large_addr = 0x123456789ABCDEF0;
    void* large_ptr = reinterpret_cast<void*>(large_addr);
    std::string large_result = PointerToHexString(large_ptr);
    EXPECT_EQ(large_result, "0x123456789abcdef0") << "Large pointer should be correctly formatted (lowercase hex)";

    // 测试零地址
    void* zero_ptr = reinterpret_cast<void*>(0);
    std::string zero_result = PointerToHexString(zero_ptr);
    EXPECT_EQ(zero_result, "0x0000000000000000") << "Zero pointer should be converted to 0x0000000000000000";
}

TEST_F(StringUtilsTest, PointerToHexStringConsistency) {
    int test_value = 100;
    void* ptr = &test_value;

    std::string result1 = PointerToHexString(ptr);
    std::string result2 = PointerToHexString(ptr);
    std::string result3 = PointerToHexString(ptr);

    EXPECT_EQ(result1, result2) << "Multiple calls should produce identical results";
    EXPECT_EQ(result2, result3) << "Multiple calls should produce identical results";
    EXPECT_EQ(result1, result3) << "Multiple calls should produce identical results";
}

TEST_F(StringUtilsTest, PointerToHexStringEdgeCases) {
    uintptr_t max_addr = ~static_cast<uintptr_t>(0);
    void* max_ptr = reinterpret_cast<void*>(max_addr);
    std::string max_result = PointerToHexString(max_ptr);

    EXPECT_EQ(max_result.substr(0, 2), "0x") << "Max pointer should start with '0x'";
    EXPECT_EQ(max_result.length(), 18) << "Max pointer result should be 18 characters long";

    for (size_t i = 2; i < max_result.length(); ++i) {
        EXPECT_EQ(max_result[i], 'f') << "Max pointer should have all 'f' characters";
    }

    void* min_ptr = reinterpret_cast<void*>(1);
    std::string min_result = PointerToHexString(min_ptr);
    EXPECT_EQ(min_result, "0x0000000000000001") << "Min non-zero pointer should be correctly formatted";
}

TEST_F(StringUtilsTest, SplitString) {
    std::vector<std::string> empty_result = SplitString("");
    EXPECT_EQ(empty_result.size(), 1);
    EXPECT_EQ(empty_result[0], "");

    std::vector<std::string> single_result = SplitString("hello");
    EXPECT_EQ(single_result.size(), 1);
    EXPECT_EQ(single_result[0], "hello");

    std::vector<std::string> multi_result = SplitString("hello,world,test");
    EXPECT_EQ(multi_result.size(), 3);
    EXPECT_EQ(multi_result[0], "hello");
    EXPECT_EQ(multi_result[1], "world");
    EXPECT_EQ(multi_result[2], "test");

    std::vector<std::string> custom_result = SplitString("hello:world:test", ':');
    EXPECT_EQ(custom_result.size(), 3);
    EXPECT_EQ(custom_result[0], "hello");
    EXPECT_EQ(custom_result[1], "world");
    EXPECT_EQ(custom_result[2], "test");

    std::vector<std::string> consecutive_result = SplitString("hello,,world");
    EXPECT_EQ(consecutive_result.size(), 3);
    EXPECT_EQ(consecutive_result[0], "hello");
    EXPECT_EQ(consecutive_result[1], "");
    EXPECT_EQ(consecutive_result[2], "world");

    std::vector<std::string> edge_result = SplitString(",hello,world,");
    EXPECT_EQ(edge_result.size(), 3);
    EXPECT_EQ(edge_result[0], "");
    EXPECT_EQ(edge_result[1], "hello");
    EXPECT_EQ(edge_result[2], "world");
}
} // namespace astate
