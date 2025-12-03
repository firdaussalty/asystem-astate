#pragma once

#include <string>

#include "discovery/service_discovery.h"

namespace astate {

class FileConfigCenter : public ConfigCenter {
 public:
    FileConfigCenter();
    explicit FileConfigCenter(const std::string& shared_path);
    ~FileConfigCenter() override;

    bool GetConfig(const std::string& key, std::string& value) override;

    bool SetConfig(const std::string& key, const std::string& value) override;

    bool RemoveConfig(const std::string& key) override;

 private:
    std::string shared_path_;
};

} // namespace astate
