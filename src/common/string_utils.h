#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace astate {

constexpr char COMMA = ',';
constexpr char COLON = ':';
constexpr char DELIMITER = COMMA;

inline std::vector<std::string> SplitString(const std::string& str, const char delimiter = DELIMITER) {
    std::vector<std::string> result;

    if (str.empty()) {
        result.emplace_back();
        return result;
    }

    std::stringstream string_stream(str);
    std::string item;
    while (std::getline(string_stream, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}

inline std::string PointerToHexString(const void* ptr) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setfill('0') << std::setw(2 * sizeof(void*)) << reinterpret_cast<uintptr_t>(ptr);
    return oss.str();
}

inline std::string ToString(const std::vector<std::string>& strings) {
    std::ostringstream oss;
    oss << "[";
    for (const auto& str : strings) {
        oss << str << ", ";
    }
    oss << "]";
    return oss.str();
}

inline std::string ToLower(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char cha) { return std::tolower(cha); });
    return lower;
}

inline std::string ToUpper(const std::string& str) {
    std::string upper = str;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char cha) { return std::toupper(cha); });
    return upper;
}

inline std::string TrimCopy(std::string_view str_view) {
    size_t b = 0;
    size_t e = str_view.size();
    while (b < e && (std::isspace(static_cast<unsigned char>(str_view[b])) != 0)) {
        ++b;
    }
    while (e > b && (std::isspace(static_cast<unsigned char>(str_view[e - 1])) != 0)) {
        --e;
    }
    return std::string(str_view.substr(b, e - b));
}

/**
 * @brief 按逗号分割，支持引号与转义
 *
 * 规则：
 * - 逗号（,）在引号外才作为分隔符；
 * - 支持双引号(")与单引号(')成对包裹，内含逗号不拆分；
 * - 支持反斜杠转义：例如 `c\,d` 会解析为 `c,d`；`\"` 与 `\'` 保留引号字符；
 * - 可选裁剪 token 前后空白；
 * - 可选保留空 token。
 */
inline std::vector<std::string>
SplitByComma(std::string_view str, bool keep_empty = false, bool respect_quotes = true, bool trim_tokens = true) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    char quote = '\0';
    bool escape = false;

    for (char cha : str) {
        if (escape) { // 上一个字符是反斜杠
            cur.push_back(cha);
            escape = false;
            continue;
        }
        if (cha == '\\') { // 进入转义
            escape = true;
            continue;
        }
        if (respect_quotes && (cha == '"' || cha == '\'')) {
            if (!in_quotes) { // 开引号
                in_quotes = true;
                quote = cha;
                continue;
            }
            if (quote == cha) { // 闭引号
                in_quotes = false;
                quote = '\0';
                continue;
            }
            // 若是不同类型引号，按普通字符处理
        }
        if (cha == ',' && !in_quotes) {
            // 切分点
            if (trim_tokens) {
                std::string tok = TrimCopy(cur);
                if (!tok.empty() || keep_empty) {
                    out.emplace_back(std::move(tok));
                }
            } else {
                if (!cur.empty() || keep_empty) {
                    out.emplace_back(std::move(cur));
                }
            }
            cur.clear();
        } else {
            cur.push_back(cha);
        }
    }

    // 收尾
    if (trim_tokens) {
        std::string tok = TrimCopy(cur);
        if (!tok.empty() || keep_empty) {
            out.emplace_back(std::move(tok));
        }
    } else {
        if (!cur.empty() || keep_empty) {
            out.emplace_back(std::move(cur));
        }
    }

    return out;
}

inline std::vector<int> SplitByCommaToInts(std::string_view str) {
    std::vector<int> vals;
    auto toks = SplitByComma(
        str,
        /*keep_empty=*/false,
        /*respect_quotes=*/false,
        /*trim_tokens=*/true);
    vals.reserve(toks.size());
    for (const auto& tok : toks) {
        vals.push_back(std::stoi(tok));
    }
    return vals;
}

} // namespace astate
