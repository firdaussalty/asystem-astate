#include "common/option.h"

#include <filesystem>
#include <fstream>

#include <spdlog/spdlog.h>

namespace astate {

void PutOptionValue(Options& options, const std::string& name, const std::string& value) {
    if (!value.empty()) {
        if (options.find(name) != options.end()) {
            SPDLOG_WARN("Option {} exists! Update it to {}.", name, value);
        }
        options[name] = value;
    } else {
        SPDLOG_INFO("Option {} is empty! Skip it.", name);
    }
}

auto GetOptionFromEnv(const std::string& name) -> std::string {
    const char* env_value = std::getenv(name.c_str());
    return env_value != nullptr ? std::string(env_value) : "";
}

void LoadOptions(Options& options) {
    std::string load_mode = GetOptionFromEnv(ASTATE_OPTIONS_LOAD_MODE);
    load_mode = load_mode.empty() ? GetOptionValue<std::string>(options, ASTATE_OPTIONS_LOAD_MODE) : load_mode;
    if (load_mode == "ENV") {
        LoadOptionsFromEnv(options);
    } else if (load_mode == "FILE") {
        LoadOptionsFromFile(options);
    } else {
        SPDLOG_ERROR("Invalid load mode: {}", load_mode);
    }
}

// Load options from ENV
void LoadOptionsFromEnv(Options& options) {
    const auto& defs = GetOptionDefinitions();

    for (const auto& [name, def] : defs) {
        const char* env_value = std::getenv(name.c_str());
        if (env_value != nullptr) {
            PutOptionValue(options, name, std::string(env_value));
        }
    }

    SPDLOG_INFO("Load options from ENV: {}", ToString(options));
}

constexpr const char* DEFAULT_ASTATE_CONFIG_FILE = "astate_config.yaml";

void LoadOptionsFromFile(Options& options, std::string file_path) {
    file_path = file_path.empty() ? GetOptionFromEnv(ASTATE_OPTIONS_FILE_PATH) : file_path;

    // If file path was not specified in ENV, try to get it from options
    file_path = file_path.empty() ? GetOptionValue<std::string>(options, ASTATE_OPTIONS_FILE_PATH) : file_path;

    // If file path was not specified in ENV or options, use default
    file_path
        = file_path.empty() ? std::filesystem::current_path().string() + "/" + DEFAULT_ASTATE_CONFIG_FILE : file_path;

    std::ifstream config_file(file_path);
    if (!config_file.is_open()) {
        SPDLOG_WARN("Config file not found: {}. Skip load options from file.", file_path);
        return;
    }

    // Start to load options from file
    std::string line;
    int line_number = 0;

    while (std::getline(config_file, line)) {
        line_number++;

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find the delimiter '='
        size_t const delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) {
            SPDLOG_WARN("Invalid config line {} in {}: {} (missing '=')", line_number, file_path, line);
            continue;
        }

        // Extract key and value
        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);

        // Trim key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        auto& defs = GetOptionDefinitions();

        // Validate key is defined in option_definitions
        if (defs.find(key) != defs.end()) {
            PutOptionValue(options, key, value);
            SPDLOG_INFO("Loaded option from file: {} = {}", key, value);
        } else {
            SPDLOG_WARN("Unknown option in config file line {}: {} (skipping)", line_number, key);
        }
    }

    config_file.close();
    SPDLOG_INFO("Load options from file [{}]: {}", file_path, ToString(options));
}

} // namespace astate
