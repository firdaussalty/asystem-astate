#include "common/option.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace astate {

class OptionTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // Clear any existing environment variables
        unsetenv(TRANSFER_ENGINE_TYPE.c_str());
        unsetenv(TRANSFER_ENGINE_META_SERVICE_ADDRESS.c_str());
        unsetenv(TRANSFER_ENGINE_LOCAL_ADDRESS.c_str());
        unsetenv(TRANSFER_ENGINE_LOCAL_PORT.c_str());
        unsetenv(TRANSFER_ENGINE_SERVICE_TYPE.c_str());
        unsetenv(TRANSFER_ENGINE_SERVICE_ADDRESS.c_str());
        unsetenv(TRANSFER_ENGINE_SERVICE_PORT.c_str());
        unsetenv(TRANSFER_ENGINE_PEERS_HOST.c_str());
    }

    void TearDown() override {
        // Clean up environment variables
        unsetenv(TRANSFER_ENGINE_TYPE.c_str());
        unsetenv(TRANSFER_ENGINE_META_SERVICE_ADDRESS.c_str());
        unsetenv(TRANSFER_ENGINE_LOCAL_ADDRESS.c_str());
        unsetenv(TRANSFER_ENGINE_LOCAL_PORT.c_str());
        unsetenv(TRANSFER_ENGINE_SERVICE_TYPE.c_str());
        unsetenv(TRANSFER_ENGINE_SERVICE_ADDRESS.c_str());
        unsetenv(TRANSFER_ENGINE_SERVICE_PORT.c_str());
        unsetenv(TRANSFER_ENGINE_PEERS_HOST.c_str());
    }
};

// Test LoadTEConfigsFromEnv with all environment variables set
TEST_F(OptionTest, LoadTEConfigsFromEnvAllVariablesSet) {
    // Set environment variables
    setenv(TRANSFER_ENGINE_TYPE.c_str(), "rdma", 1);
    setenv(TRANSFER_ENGINE_META_SERVICE_ADDRESS.c_str(), "192.168.1.100:8080", 1);
    setenv(TRANSFER_ENGINE_LOCAL_ADDRESS.c_str(), "192.168.1.101", 1);
    setenv(TRANSFER_ENGINE_LOCAL_PORT.c_str(), "8081", 1);
    setenv(TRANSFER_ENGINE_SERVICE_TYPE.c_str(), "http", 1);
    setenv(TRANSFER_ENGINE_SERVICE_ADDRESS.c_str(), "192.168.1.102", 1);
    setenv(TRANSFER_ENGINE_SERVICE_PORT.c_str(), "8082", 1);
    setenv(TRANSFER_ENGINE_PEERS_HOST.c_str(), "192.168.1.103,192.168.1.104", 1);

    Options options;
    LoadOptionsFromEnv(options);

    // Verify all values are loaded correctly
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "rdma");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_META_SERVICE_ADDRESS), "192.168.1.100:8080");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS), "192.168.1.101");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 8081);
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_TYPE), "http");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_ADDRESS), "192.168.1.102");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_SERVICE_PORT), 8082);

    auto peer_hosts = GetOptionValue<std::vector<std::string>>(options, "TRANSFER_ENGINE_PEERS_HOST");
    EXPECT_EQ(peer_hosts.size(), 2);
    EXPECT_EQ(peer_hosts[0], "192.168.1.103");
    EXPECT_EQ(peer_hosts[1], "192.168.1.104");
}

// Test LoadTEConfigsFromEnv with no environment variables set
TEST_F(OptionTest, LoadTEConfigsFromEnvNoVariablesSet) {
    Options options;
    LoadOptionsFromEnv(options);

    // Verify all configs are empty
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_META_SERVICE_ADDRESS), "");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS), "");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 0);
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_TYPE), "");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_ADDRESS), "");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_SERVICE_PORT), 0);
    EXPECT_EQ(
        GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST), std::vector<std::string>{});
}

// Test LoadTEConfigsFromEnv with partial environment variables set
TEST_F(OptionTest, LoadTEConfigsFromEnvPartialVariablesSet) {
    // Set only some environment variables
    setenv("TRANSFER_ENGINE_TYPE", "tcp", 1);
    setenv("TRANSFER_ENGINE_LOCAL_ADDRESS", "127.0.0.1", 1);
    setenv("TRANSFER_ENGINE_LOCAL_PORT", "9000", 1);

    Options options;
    LoadOptionsFromEnv(options);

    // Verify only set variables are loaded
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "tcp");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS), "127.0.0.1");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 9000);

    // Verify all configs are empty
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_META_SERVICE_ADDRESS), "");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_TYPE), "");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_ADDRESS), "");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_SERVICE_PORT), 0);
    EXPECT_EQ(
        GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST), std::vector<std::string>{});
}

// Test GetTransferEngineType
TEST_F(OptionTest, GetTransferEngineType) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_TYPE, "rdma");

    auto result = GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE);
    EXPECT_EQ(result, "rdma");
}

// Test GetTransferEngineType with missing key
TEST_F(OptionTest, GetTransferEngineTypeMissingKey) {
    Options options;

    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "");
}

// Test GetTransferEngineMetaServiceAddress
TEST_F(OptionTest, GetTransferEngineMetaServiceAddress) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_META_SERVICE_ADDRESS, "192.168.1.100:8080");

    auto result = GetOptionValue<std::string>(options, TRANSFER_ENGINE_META_SERVICE_ADDRESS);
    EXPECT_EQ(result, "192.168.1.100:8080");
}

// Test GetTransferEngineMetaServiceAddress with missing key
TEST_F(OptionTest, GetTransferEngineMetaServiceAddressMissingKey) {
    Options options;

    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_META_SERVICE_ADDRESS), "");
}

// Test GetTransferEngineAddress
TEST_F(OptionTest, GetTransferEngineAddress) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_LOCAL_ADDRESS, "192.168.1.101");

    auto result = GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS);
    EXPECT_EQ(result, "192.168.1.101");
}

// Test GetTransferEngineAddress with missing key
TEST_F(OptionTest, GetTransferEngineAddressMissingKey) {
    Options options;

    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS), "");
}

// Test GetTransferEnginePort with string value
TEST_F(OptionTest, GetTransferEnginePortStringValue) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_LOCAL_PORT, "8081");

    int result = GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT);
    EXPECT_EQ(result, 8081);
}

// Test GetTransferEngineServiceTYPE
TEST_F(OptionTest, GetTransferEngineServiceType) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_SERVICE_TYPE, "rpc");

    auto result = GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_TYPE);
    EXPECT_EQ(result, "rpc");
}

// Test GetTransferEngineServiceType with missing key
TEST_F(OptionTest, GetTransferEngineServiceTypeMissingKey) {
    Options options;

    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_TYPE), "");
}

// Test GetTransferEngineServiceAddress
TEST_F(OptionTest, GetTransferEngineServiceAddress) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_SERVICE_ADDRESS, "192.168.1.102");

    auto result = GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_ADDRESS);
    EXPECT_EQ(result, "192.168.1.102");
}

// Test GetTransferEngineServicePort with string value
TEST_F(OptionTest, GetTransferEngineServicePortStringValue) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_SERVICE_PORT, "8082");

    auto result = GetOptionValue<int>(options, TRANSFER_ENGINE_SERVICE_PORT);
    EXPECT_EQ(result, 8082);
}

// Test GetTransferEngineServicePeersHost with comma-separated string
TEST_F(OptionTest, GetTransferEngineServicePeersHostCommaSeparated) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_PEERS_HOST, "192.168.1.103,192.168.1.104,192.168.1.105");

    auto result = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST);

    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "192.168.1.103");
    EXPECT_EQ(result[1], "192.168.1.104");
    EXPECT_EQ(result[2], "192.168.1.105");
}

// Test GetTransferEngineServicePeersHost with single host
TEST_F(OptionTest, GetTransferEngineServicePeersHostSingleHost) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_PEERS_HOST, "192.168.1.103");

    auto result = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST);

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], "192.168.1.103");
}

// Test GetTransferEngineServicePeersHost with empty string
TEST_F(OptionTest, GetTransferEngineServicePeersHostEmptyString) {
    Options options;
    PutOptionValue(options, TRANSFER_ENGINE_PEERS_HOST, "");

    auto result = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST);

    // When splitting an empty string, we should get a vector with one empty string
    EXPECT_EQ(result.size(), 0);
}

// Test Config stream operator
TEST_F(OptionTest, ConfigStreamOperator) {
    Options options;
    PutOptionValue(options, "key1", "value1");
    PutOptionValue(options, "key2", "value2");

    std::ostringstream oss;
    oss << options;

    std::string result = oss.str();
    EXPECT_TRUE(result.find("key1: value1") != std::string::npos);
    EXPECT_TRUE(result.find("key2: value2") != std::string::npos);
    EXPECT_TRUE(result.find('{') != std::string::npos);
    EXPECT_TRUE(result.find('}') != std::string::npos);
}

// Test Options stream operator with empty options
TEST_F(OptionTest, OptionsStreamOperatorEmptyOption) {
    Options options;

    std::ostringstream oss;
    oss << options;

    std::string result = oss.str();
    EXPECT_EQ(result, "{}");
}

// Test Optino stream operator with single entry
TEST_F(OptionTest, OptionStreamOperatorSingleEntry) {
    Options options;
    PutOptionValue(options, "key1", "value1");

    std::ostringstream oss;
    oss << options;

    std::string result = oss.str();
    EXPECT_EQ(result, "{key1: value1}");
}

// Integration test: Load from env and get values
TEST_F(OptionTest, IntegrationTestLoadFromEnvAndGetValues) {
    // Set environment variables
    setenv(TRANSFER_ENGINE_TYPE.c_str(), "rdma", 1);
    setenv(TRANSFER_ENGINE_LOCAL_ADDRESS.c_str(), "192.168.1.101", 1);
    setenv(TRANSFER_ENGINE_LOCAL_PORT.c_str(), "8081", 1);
    setenv(TRANSFER_ENGINE_PEERS_HOST.c_str(), "host1,host2,host3", 1);

    Options options;
    LoadOptionsFromEnv(options);

    // Test getting values
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 8081);
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "rdma");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS), "192.168.1.101");

    auto peers = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST);
    EXPECT_EQ(peers.size(), 3);
    EXPECT_EQ(peers[0], "host1");
    EXPECT_EQ(peers[1], "host2");
    EXPECT_EQ(peers[2], "host3");
}

// Test ParseBool function
TEST_F(OptionTest, ParseBoolValidStringValues) {
    // Test string values that should parse to true
    EXPECT_TRUE(ParseBool(std::make_any<std::string>("true")));
    EXPECT_TRUE(ParseBool(std::make_any<std::string>("True")));
    EXPECT_TRUE(ParseBool(std::make_any<std::string>("TRUE")));
    EXPECT_TRUE(ParseBool(std::make_any<std::string>("1")));

    // Test string values that should parse to false
    EXPECT_FALSE(ParseBool(std::make_any<std::string>("false")));
    EXPECT_FALSE(ParseBool(std::make_any<std::string>("False")));
    EXPECT_FALSE(ParseBool(std::make_any<std::string>("FALSE")));
    EXPECT_FALSE(ParseBool(std::make_any<std::string>("0")));
}

TEST_F(OptionTest, ParseBoolValidBoolValues) {
    // Test bool values
    EXPECT_TRUE(ParseBool(std::make_any<bool>(true)));
    EXPECT_FALSE(ParseBool(std::make_any<bool>(false)));
}

TEST_F(OptionTest, ParseBoolInvalidStringValues) {
    // Test invalid string values
    EXPECT_THROW(ParseBool(std::make_any<std::string>("invalid")), std::invalid_argument);
    EXPECT_THROW(ParseBool(std::make_any<std::string>("yes")), std::invalid_argument);
    EXPECT_THROW(ParseBool(std::make_any<std::string>("no")), std::invalid_argument);
    EXPECT_THROW(ParseBool(std::make_any<std::string>("2")), std::invalid_argument);
    EXPECT_THROW(ParseBool(std::make_any<std::string>("")), std::invalid_argument);
}

TEST_F(OptionTest, ParseBoolInvalidTypes) {
    // Test invalid types
    EXPECT_THROW(ParseBool(std::make_any<int>(1)), std::invalid_argument);
    EXPECT_THROW(ParseBool(std::make_any<float>(1.0F)), std::invalid_argument);
    EXPECT_THROW(ParseBool(std::make_any<double>(1.0)), std::invalid_argument);
}

// Test ParseFloat function
TEST_F(OptionTest, ParseFloatValidStringValues) {
    // Test valid string values
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<std::string>("3.14")), 3.14F);
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<std::string>("0.0")), 0.0F);
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<std::string>("-2.5")), -2.5F);
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<std::string>("123")), 123.0F);
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<std::string>("1e-3")), 1e-3F);
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<std::string>("3.14.15")), 3.14F);
}

TEST_F(OptionTest, ParseFloatValidFloatValues) {
    // Test float values
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<float>(3.14F)), 3.14F);
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<float>(0.0F)), 0.0F);
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<float>(-2.5F)), -2.5F);
}

TEST_F(OptionTest, ParseFloatInvalidStringValues) {
    // Test invalid string values
    EXPECT_THROW(ParseFloat(std::make_any<std::string>("invalid")), std::invalid_argument);
    EXPECT_THROW(ParseFloat(std::make_any<std::string>("abc123")), std::invalid_argument);
    EXPECT_THROW(ParseFloat(std::make_any<std::string>("")), std::invalid_argument);
}

TEST_F(OptionTest, ParseFloatInvalidTypes) {
    // Test invalid types
    EXPECT_THROW(ParseFloat(std::make_any<bool>(true)), std::invalid_argument);
    EXPECT_THROW(ParseFloat(std::make_any<int>(123)), std::invalid_argument);
}

// Test ParseDouble function
TEST_F(OptionTest, ParseDoubleValidStringValues) {
    // Test valid string values
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<std::string>("3.141592653589793")), 3.141592653589793);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<std::string>("0.0")), 0.0);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<std::string>("-2.5")), -2.5);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<std::string>("123")), 123.0);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<std::string>("1e-10")), 1e-10);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<std::string>("3.14.15")), 3.14);
}

TEST_F(OptionTest, ParseDoubleValidDoubleValues) {
    // Test double values
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<double>(3.141592653589793)), 3.141592653589793);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<double>(0.0)), 0.0);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<double>(-2.5)), -2.5);
}

TEST_F(OptionTest, ParseDoubleInvalidStringValues) {
    // Test invalid string values
    EXPECT_THROW(ParseDouble(std::make_any<std::string>("invalid")), std::invalid_argument);
    EXPECT_THROW(ParseDouble(std::make_any<std::string>("abc123")), std::invalid_argument);
    EXPECT_THROW(ParseDouble(std::make_any<std::string>("")), std::invalid_argument);
}

TEST_F(OptionTest, ParseDoubleInvalidTypes) {
    // Test invalid types
    EXPECT_THROW(ParseDouble(std::make_any<bool>(true)), std::invalid_argument);
    EXPECT_THROW(ParseDouble(std::make_any<int>(123)), std::invalid_argument);
}

// Integration test for all parsers with edge cases
TEST_F(OptionTest, ParseFunctionEdgeCases) {
    // Test ParseBool with edge case strings
    EXPECT_TRUE(ParseBool(std::make_any<std::string>("1")));
    EXPECT_FALSE(ParseBool(std::make_any<std::string>("0")));

    // Test ParseFloat with edge values
    EXPECT_TRUE(std::isinf(ParseFloat(std::make_any<std::string>("inf"))));
    EXPECT_TRUE(std::isnan(ParseFloat(std::make_any<std::string>("nan"))));

    // Test ParseDouble with edge values
    EXPECT_TRUE(std::isinf(ParseDouble(std::make_any<std::string>("inf"))));
    EXPECT_TRUE(std::isnan(ParseDouble(std::make_any<std::string>("nan"))));

    // Test very large and small values
    EXPECT_FLOAT_EQ(ParseFloat(std::make_any<std::string>("1e30")), 1e30F);
    EXPECT_DOUBLE_EQ(ParseDouble(std::make_any<std::string>("1e-100")), 1e-100);
}

// Test LoadOptionsFromFile with valid config file
TEST_F(OptionTest, LoadOptionsFromFileValidConfigFile) {
    // Create a temporary config file
    std::string config_content = "# This is a comment\n"
                                 "TRANSFER_ENGINE_TYPE=rdma\n"
                                 "TRANSFER_ENGINE_LOCAL_ADDRESS=192.168.1.100\n"
                                 "TRANSFER_ENGINE_LOCAL_PORT=8080\n"
                                 "\n" // Empty line
                                 "  TRANSFER_ENGINE_SERVICE_TYPE  =  http  \n" // Test whitespace trimming
                                 "TRANSFER_ENGINE_PEERS_HOST=host1,host2,host3\n";

    std::ofstream config_file("test_config.txt");
    config_file << config_content;
    config_file.close();

    // Set environment variable to use our test config file
    setenv("ASTATE_OPTIONS_FILE_PATH", "test_config.txt", 1);

    Options options;
    LoadOptionsFromFile(options);

    // Verify loaded values
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "rdma");
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_LOCAL_ADDRESS), "192.168.1.100");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 8080);
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_SERVICE_TYPE), "http");

    auto peer_hosts = GetOptionValue<std::vector<std::string>>(options, TRANSFER_ENGINE_PEERS_HOST);
    EXPECT_EQ(peer_hosts.size(), 3);
    EXPECT_EQ(peer_hosts[0], "host1");
    EXPECT_EQ(peer_hosts[1], "host2");
    EXPECT_EQ(peer_hosts[2], "host3");

    // Clean up
    std::remove("test_config.txt");
    unsetenv("ASTATE_OPTIONS_FILE_PATH");
}

// Test LoadOptionsFromFile with non-existent file
TEST_F(OptionTest, LoadOptionsFromFileNonExistentFile) {
    // Set environment variable to a non-existent file
    setenv("ASTATE_OPTIONS_FILE_PATH", "non_existent_config.txt", 1);

    Options options;
    LoadOptionsFromFile(options);

    // Should load no options (file doesn't exist)
    EXPECT_TRUE(options.empty());

    // Clean up
    unsetenv("ASTATE_OPTIONS_FILE_PATH");
}

// Test LoadOptionsFromFile with invalid config lines
TEST_F(OptionTest, LoadOptionsFromFileInvalidConfigLines) {
    // Create a config file with invalid lines
    std::string config_content = "TRANSFER_ENGINE_TYPE=rdma\n"
                                 "INVALID_LINE_WITHOUT_EQUALS\n" // Invalid line
                                 "TRANSFER_ENGINE_LOCAL_PORT=8080\n"
                                 "UNKNOWN_OPTION=value\n" // Unknown option
                                 "=EMPTY_KEY\n" // Empty key
                                 "EMPTY_VALUE=\n"; // Empty value

    std::ofstream config_file("test_invalid_config.txt");
    config_file << config_content;
    config_file.close();

    // Set environment variable to use our test config file
    setenv("ASTATE_OPTIONS_FILE_PATH", "test_invalid_config.txt", 1);

    Options options;
    LoadOptionsFromFile(options);

    // Verify only valid options are loaded
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "rdma");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 8080);

    // Invalid options should not be loaded
    EXPECT_EQ(options.find("INVALID_LINE_WITHOUT_EQUALS"), options.end());
    EXPECT_EQ(options.find("UNKNOWN_OPTION"), options.end());

    // Clean up
    std::remove("test_invalid_config.txt");
    unsetenv("ASTATE_OPTIONS_FILE_PATH");
}

// Test LoadOptionsFromFile with default file path
TEST_F(OptionTest, LoadOptionsFromFileDefaultFilePath) {
    // Create default config file
    std::string config_content = "TRANSFER_ENGINE_TYPE=tcp\n";

    std::ofstream config_file("astate_config.yaml");
    config_file << config_content;
    config_file.close();

    // Don't set environment variable, should use default path
    Options options;
    LoadOptionsFromFile(options, "astate_config.yaml");

    // Verify loaded value
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "tcp");

    // Clean up
    std::remove("astate_config.yaml");
}

// Test LoadOptionsFromFile with various data types
TEST_F(OptionTest, LoadOptionsFromFileVariousDataTypes) {
    // Create a config file with different data types
    std::string config_content = "TRANSFER_ENGINE_TYPE=rdma\n"
                                 "TRANSFER_ENGINE_LOCAL_PORT=9000\n"
                                 "TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY=true\n"
                                 "TRANSFER_ENGINE_SERVICE_FIXED_PORT=false\n";

    std::ofstream config_file("test_types_config.txt");
    config_file << config_content;
    config_file.close();

    // Set environment variable to use our test config file
    setenv("ASTATE_OPTIONS_FILE_PATH", "test_types_config.txt", 1);

    Options options;
    LoadOptionsFromFile(options);

    // Verify loaded values with correct types
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "rdma");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 9000);
    EXPECT_EQ(GetOptionValue<bool>(options, "TRANSFER_ENGINE_SERVICE_SKIP_DISCOVERY"), true);
    EXPECT_EQ(GetOptionValue<bool>(options, "TRANSFER_ENGINE_SERVICE_FIXED_PORT"), false);

    // Clean up
    std::remove("test_types_config.txt");
    unsetenv("ASTATE_OPTIONS_FILE_PATH");
}

// Integration test: LoadOptions with FILE mode
TEST_F(OptionTest, LoadOptionsFileMode) {
    // Create a config file
    std::string config_content = "TRANSFER_ENGINE_TYPE=rdma\n"
                                 "TRANSFER_ENGINE_LOCAL_PORT=7777\n";

    std::ofstream config_file("integration_test_config.txt");
    config_file << config_content;
    config_file.close();

    // Set environment variables for FILE mode
    setenv("ASTATE_OPTIONS_LOAD_MODE", "FILE", 1);
    setenv("ASTATE_OPTIONS_FILE_PATH", "integration_test_config.txt", 1);

    Options options;
    LoadOptions(options);

    // Verify loaded values
    EXPECT_EQ(GetOptionValue<std::string>(options, TRANSFER_ENGINE_TYPE), "rdma");
    EXPECT_EQ(GetOptionValue<int>(options, TRANSFER_ENGINE_LOCAL_PORT), 7777);

    // Clean up
    std::remove("integration_test_config.txt");
    unsetenv("ASTATE_OPTIONS_LOAD_MODE");
    unsetenv("ASTATE_OPTIONS_FILE_PATH");
}

} // namespace astate
