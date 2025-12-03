#include "discovery/file_config_center.h"

#include <filesystem>
#include <fstream>

namespace astate {

FileConfigCenter::FileConfigCenter() {
    // Retrieve the shared path from the environment variable.
    // shared_path is set by the environment variable ASTATE_SHARED_PATH and CLUSTER_NAME.
    const char* env_str = std::getenv("ASTATE_SHARED_PATH");
    if (env_str == nullptr) {
        throw std::runtime_error("ASTATE_SHARED_PATH is not set");
    }
    std::string astate_shared_path(env_str);

    env_str = std::getenv("CLUSTER_NAME");
    if (env_str == nullptr) {
        throw std::runtime_error("CLUSTER_NAME is not set");
    }
    std::string job_name(env_str);

    shared_path_ = astate_shared_path + "/" + job_name + "_ASTATE_CONFIG";
    if (!std::filesystem::exists(shared_path_)) {
        std::filesystem::create_directories(shared_path_);
    }
    SPDLOG_INFO("FileConfigCenter initialized with shared_path: {}", shared_path_);
}

FileConfigCenter::FileConfigCenter(const std::string& shared_path) {
    if (shared_path.empty()) {
        throw std::invalid_argument("shared_path is empty");
    }

    if (!std::filesystem::exists(shared_path)) {
        std::filesystem::create_directories(shared_path);
    }

    shared_path_ = shared_path;
    SPDLOG_INFO("FileConfigCenter initialized with shared_path: {}", shared_path_);
}

FileConfigCenter::~FileConfigCenter() {
    // Notice: All config files are removed when the config center is destroyed.
    // Currently, the config center is only used for the service discovery during initialization, if any persistent
    // config is needed, the file config center should be updated.
    for (const auto& entry : std::filesystem::directory_iterator(shared_path_)) {
        std::filesystem::remove(entry.path());
    }
    std::filesystem::remove(shared_path_);
    SPDLOG_INFO("FileConfigCenter destroyed");
}

bool FileConfigCenter::GetConfig(const std::string& key, std::string& value) {
    if (key.empty()) {
        SPDLOG_ERROR("Key is empty");
        return false;
    }

    std::string config_file_path = shared_path_ + "/" + key;
    if (!std::filesystem::exists(config_file_path)) {
        SPDLOG_WARN("Config file not found");
        return false;
    }

    std::ifstream ifs(config_file_path);
    if (!ifs.is_open()) {
        SPDLOG_ERROR("Failed to open config file");
        return false;
    }

    std::getline(ifs, value);
    ifs.close();
    return true;
}

bool FileConfigCenter::SetConfig(const std::string& key, const std::string& value) {
    std::string config_file_path = shared_path_ + "/" + key;
    std::ofstream ofs(config_file_path);

    if (!ofs.is_open()) {
        SPDLOG_ERROR("Failed to open/create config file");
        return false;
    }

    ofs << value;
    ofs.close();
    return true;
}

bool FileConfigCenter::RemoveConfig(const std::string& key) {
    if (key.empty()) {
        SPDLOG_WARN("Key to be removed is empty");
        return true;
    }

    std::string config_file_path = shared_path_ + "/" + key;
    if (!std::filesystem::exists(config_file_path)) {
        SPDLOG_WARN("Config file not found");
    }

    std::filesystem::remove(config_file_path);

    // make sure again after remove operation, whether the config file is removed successfully.
    if (std::filesystem::exists(config_file_path)) {
        SPDLOG_ERROR("Failed to remove config file");
        return false;
    }
    return true;
}

} // namespace astate
