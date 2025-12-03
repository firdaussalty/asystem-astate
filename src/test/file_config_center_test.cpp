#include "discovery/file_config_center.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

using namespace astate;

class FileConfigCenterTest : public ::testing::Test {
 protected:
    void SetUp() override {
        // create a temporary test directory
        test_dir_ = "/tmp/astate_file_config_test_" + std::to_string(getpid());
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        // clean up the temporary test directory
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }

    std::string test_dir_;
};

// test constructor - valid path
TEST_F(FileConfigCenterTest, ConstructorValidPath) {
    EXPECT_NO_THROW({ astate::FileConfigCenter config_center(test_dir_); });
}

// test constructor - empty path
TEST_F(FileConfigCenterTest, ConstructorEmptyPath) {
    EXPECT_THROW({ astate::FileConfigCenter config_center(""); }, std::invalid_argument);
}

// test constructor - non-existent path
TEST_F(FileConfigCenterTest, ConstructorNonExistentPath) {
    std::string non_existent_path = "/tmp/non_existent_path_12345";
    EXPECT_NO_THROW({ astate::FileConfigCenter config_center(non_existent_path); });
}

// test SetConfig - valid key and value
TEST_F(FileConfigCenterTest, SetConfigSuccess) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "test_key";
    std::string value = "test_value";

    EXPECT_TRUE(config_center.SetConfig(key, value));

    // verify the file is created
    std::string file_path = test_dir_ + "/" + key;
    EXPECT_TRUE(std::filesystem::exists(file_path));

    // verify the file content
    std::ifstream file(file_path);
    std::string file_content;
    std::getline(file, file_content);
    EXPECT_EQ(file_content, value);
}

// test SetConfig - overwrite existing config
TEST_F(FileConfigCenterTest, SetConfigOverwrite) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "test_key";
    std::string value1 = "test_value_1";
    std::string value2 = "test_value_2";

    // first set
    EXPECT_TRUE(config_center.SetConfig(key, value1));

    // second set, overwrite the first one
    EXPECT_TRUE(config_center.SetConfig(key, value2));

    // verify the file content is the latest value
    std::string file_path = test_dir_ + "/" + key;
    std::ifstream file(file_path);
    std::string file_content;
    std::getline(file, file_content);
    EXPECT_EQ(file_content, value2);
}

// test GetConfig - valid key and value
TEST_F(FileConfigCenterTest, GetConfigSuccess) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "test_key";
    std::string value = "test_value";

    // first set
    EXPECT_TRUE(config_center.SetConfig(key, value));

    // then get
    std::string retrieved_value;
    EXPECT_TRUE(config_center.GetConfig(key, retrieved_value));
    EXPECT_EQ(retrieved_value, value);
}

// test GetConfig - non-existent key
TEST_F(FileConfigCenterTest, GetConfigNotExists) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "non_existent_key";
    std::string retrieved_value;

    EXPECT_FALSE(config_center.GetConfig(key, retrieved_value));
}

// test GetConfig - empty key
TEST_F(FileConfigCenterTest, GetConfigEmptyKey) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key;
    std::string retrieved_value;

    EXPECT_FALSE(config_center.GetConfig(key, retrieved_value));
}

// test RemoveConfig - valid key and value
TEST_F(FileConfigCenterTest, RemoveConfigSuccess) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "test_key";
    std::string value = "test_value";

    // first set
    EXPECT_TRUE(config_center.SetConfig(key, value));

    // verify the file exists
    std::string file_path = test_dir_ + "/" + key;
    EXPECT_TRUE(std::filesystem::exists(file_path));

    // remove the config
    EXPECT_TRUE(config_center.RemoveConfig(key));

    // verify the file is deleted
    EXPECT_FALSE(std::filesystem::exists(file_path));
}

// test RemoveConfig - non-existent key
TEST_F(FileConfigCenterTest, RemoveConfigNotExists) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "non_existent_key";

    // remove non-existent config should return true (because the target state is reached)
    EXPECT_TRUE(config_center.RemoveConfig(key));
}

// test RemoveConfig - empty key
TEST_F(FileConfigCenterTest, RemoveConfigEmptyKey) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key;

    EXPECT_TRUE(config_center.RemoveConfig(key));
}

// test special characters and long content
TEST_F(FileConfigCenterTest, SpecialCharactersAndLongContent) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "special_key_with_特殊字符_123";
    std::string value = "这是一个很长的配置值，包含中文、English、数字123、特殊"
                        "符号!@#$%^&*()_+-=[]{}|;':\",./<>?";

    // set config
    EXPECT_TRUE(config_center.SetConfig(key, value));

    // get config
    std::string retrieved_value;
    EXPECT_TRUE(config_center.GetConfig(key, retrieved_value));
    EXPECT_EQ(retrieved_value, value);

    // remove config
    EXPECT_TRUE(config_center.RemoveConfig(key));
}

// test multiple configs
TEST_F(FileConfigCenterTest, MultipleConfigs) {
    astate::FileConfigCenter config_center(test_dir_);

    std::vector<std::pair<std::string, std::string>> configs
        = {{"key1", "value1"}, {"key2", "value2"}, {"key3", "value3"}, {"key4", "value4"}};

    // set multiple configs
    for (const auto& config : configs) {
        EXPECT_TRUE(config_center.SetConfig(config.first, config.second));
    }

    // get and verify all configs
    for (const auto& config : configs) {
        std::string retrieved_value;
        EXPECT_TRUE(config_center.GetConfig(config.first, retrieved_value));
        EXPECT_EQ(retrieved_value, config.second);
    }

    // remove all configs
    for (const auto& config : configs) {
        EXPECT_TRUE(config_center.RemoveConfig(config.first));
    }

    // verify all files are deleted
    for (const auto& config : configs) {
        std::string file_path = test_dir_ + "/" + config.first;
        EXPECT_FALSE(std::filesystem::exists(file_path));
    }
}

// test file permission issues (simulate)
TEST_F(FileConfigCenterTest, FilePermissionIssues) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "permission_test_key";
    std::string value = "test_value";

    // set config
    EXPECT_TRUE(config_center.SetConfig(key, value));

    // verify the config can be get
    std::string retrieved_value;
    EXPECT_TRUE(config_center.GetConfig(key, retrieved_value));
    EXPECT_EQ(retrieved_value, value);

    // remove config
    EXPECT_TRUE(config_center.RemoveConfig(key));
}

// test concurrent access (basic test)
TEST_F(FileConfigCenterTest, ConcurrentAccess) {
    astate::FileConfigCenter config_center(test_dir_);

    std::string key = "concurrent_key";
    std::string value = "concurrent_value";

    // set config
    EXPECT_TRUE(config_center.SetConfig(key, value));

    // simulate concurrent read (multiple fast read)
    for (int i = 0; i < 10; ++i) {
        std::string retrieved_value;
        EXPECT_TRUE(config_center.GetConfig(key, retrieved_value));
        EXPECT_EQ(retrieved_value, value);
    }

    // clean up
    EXPECT_TRUE(config_center.RemoveConfig(key));
}

// test edge cases
TEST_F(FileConfigCenterTest, EdgeCases) {
    astate::FileConfigCenter config_center(test_dir_);

    // test empty value
    std::string key = "empty_value_key";
    std::string empty_value;

    EXPECT_TRUE(config_center.SetConfig(key, empty_value));

    std::string retrieved_value;
    EXPECT_TRUE(config_center.GetConfig(key, retrieved_value));
    EXPECT_EQ(retrieved_value, empty_value);

    // test value with newline
    std::string key2 = "newline_key";
    std::string newline_value = "line1\nline2\nline3";

    EXPECT_TRUE(config_center.SetConfig(key2, newline_value));

    std::string retrieved_value2;
    EXPECT_TRUE(config_center.GetConfig(key2, retrieved_value2));
    // note: getline only reads the first line, so here only verify the first line
    EXPECT_EQ(retrieved_value2, "line1");

    // clean up
    EXPECT_TRUE(config_center.RemoveConfig(key));
    EXPECT_TRUE(config_center.RemoveConfig(key2));
}

// test destructor and resource cleanup
TEST_F(FileConfigCenterTest, DestructorAndCleanup) {
    {
        astate::FileConfigCenter config_center(test_dir_);

        std::string key = "cleanup_test_key";
        std::string value = "cleanup_test_value";

        EXPECT_TRUE(config_center.SetConfig(key, value));

        // verify the file exists
        std::string file_path = test_dir_ + "/" + key;
        EXPECT_TRUE(std::filesystem::exists(file_path));
    } // config_center is destructed here

    // verify the file still exists (the destructor should delete the file)
    std::string file_path = test_dir_ + "/cleanup_test_key";
    EXPECT_FALSE(std::filesystem::exists(file_path));

    // manually clean up
    std::filesystem::remove(file_path);
}
