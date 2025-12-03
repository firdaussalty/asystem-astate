#!/bin/bash

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}项目根目录: $PROJECT_ROOT${NC}"

# 检查工具函数
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}错误: 未找到 $1，请先安装${NC}"
        return 1
    fi
    return 0
}

# 检查必要的工具
check_tool clang-format || exit 1
check_tool clang-tidy || exit 1

# 检查配置文件
if [ ! -f "$PROJECT_ROOT/.clang-format" ]; then
    echo -e "${RED}错误: 在项目根目录未找到 .clang-format 文件${NC}"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/.clang-tidy" ]; then
    echo -e "${RED}错误: 在项目根目录未找到 .clang-tidy 文件${NC}"
    exit 1
fi

# 查找所有 C++ 文件
find_cpp_files() {
    find astate_cache -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) \
        -not -path "*/build/*" \
        -not -path "*/thirdparties/*" \
        -not -path "*/.cache/*"
}

# 格式化检查
check_format() {
    echo -e "${YELLOW}开始检查代码格式...${NC}"
    local has_errors=0

    while IFS= read -r -d '' file; do
        if ! clang-format --Werror --dry-run "$file" &>/dev/null; then
            echo -e "${RED}格式错误: $file${NC}"
            has_errors=1
        fi
    done < <(find_cpp_files | tr '\n' '\0')

    if [ $has_errors -eq 0 ]; then
        echo -e "${GREEN}✓ 代码格式检查通过${NC}"
        return 0
    else
        echo -e "${RED}✗ 代码格式检查未通过，请手动调整或使用 --fix 选项自动修复${NC}"
        return 1
    fi
}

# 格式化修复
fix_format() {
    echo -e "${YELLOW}开始自动修复代码格式...${NC}"
    local file_count=0

    while IFS= read -r -d '' file; do
        echo "格式化: $file"
        clang-format -i "$file"
        ((file_count++))
    done < <(find_cpp_files | tr '\n' '\0')
    
    echo -e "${GREEN}✓ 已格式化 $file_count 个文件${NC}"
}

# 静态检查
check_tidy() {
    echo -e "${YELLOW}开始静态代码检查...${NC}"

    # 检查是否有编译数据库
    if [ ! -f "$PROJECT_ROOT/astate_cache/build/compile_commands.json" ]; then
        echo -e "${YELLOW}警告: 未找到编译数据库，clang-tidy 可能无法正常工作${NC}"
        echo -e "${YELLOW}请使用 CMake 生成: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ...${NC}"
    fi

    local has_errors=0
    local temp_file=$(mktemp)

    while IFS= read -r -d '' file; do
        if ! clang-tidy -p "$PROJECT_ROOT/astate_cache/build" "$file" --warnings-as-errors='*' > "$temp_file" 2>&1; then
            echo -e "${RED}静态检查错误: $file${NC}"
            cat "$temp_file"
            has_errors=1
        fi
    done < <(find_cpp_files | tr '\n' '\0')

    rm -f "$temp_file"

    if [ $has_errors -eq 0 ]; then
        echo -e "${GREEN}✓ 静态检查通过${NC}"
        return 0
    else
        echo -e "${RED}✗ 静态检查未通过，请手动调整或使用 --fix 选项自动修复${NC}"
        return 1
    fi
}

# 静态修复
fix_tidy() {
    echo -e "${YELLOW}开始自动修复静态检查问题...${NC}"

    if [ ! -f "$PROJECT_ROOT/astate_cache/build/compile_commands.json" ]; then
        echo -e "${RED}错误: 未找到编译数据库，无法进行静态修复${NC}"
        echo -e "${YELLOW}请使用 CMake 生成: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ...${NC}"
        return 1
    fi

    local file_count=0

    while IFS= read -r -d '' file; do
        echo "静态修复: $file"
        if clang-tidy -p "$PROJECT_ROOT/astate_cache/build" "$file" -fix; then
            ((file_count++))
        else
            echo -e "${YELLOW}警告: $file 修复可能不完整${NC}"
        fi
    done < <(find_cpp_files | tr '\n' '\0')

    echo -e "${GREEN}✓ 已尝试修复 $file_count 个文件${NC}"
    echo -e "${YELLOW}注意: 某些问题可能需要手动修复${NC}"
}

# 显示帮助信息
show_help() {
    echo "代码格式化和静态检查工具"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --check        只进行检查（默认行为）"
    echo "  --fix          自动修复问题"
    echo "  --format-only  只进行格式相关操作"
    echo "  --tidy-only    只进行静态检查相关操作"
    echo "  --help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 检查格式和静态问题"
    echo "  $0 --fix              # 自动修复格式和静态问题"
    echo "  $0 --format-only      # 只检查格式"
    echo "  $0 --tidy-only --fix  # 只修复静态问题"
}

# 解析命令行参数
MODE="check"  # 默认模式：check
TARGET="all"  # 默认目标：all

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            MODE="check"
            shift
            ;;
        --fix)
            MODE="fix"
            shift
            ;;
        --format-only)
            TARGET="format"
            shift
            ;;
        --tidy-only)
            TARGET="tidy"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 执行相应操作
case "$MODE-$TARGET" in
    "check-all")
        check_format
        format_result=$?
        check_tidy
        tidy_result=$?
        exit $((format_result + tidy_result))
        ;;
    "check-format")
        check_format
        exit $?
        ;;
    "check-tidy")
        check_tidy
        exit $?
        ;;
    "fix-all")
        fix_format
        fix_tidy
        echo -e "${GREEN}✓ 所有自动修复已完成${NC}"
        ;;
    "fix-format")
        fix_format
        ;;
    "fix-tidy")
        fix_tidy
        ;;
    *)
        echo -e "${RED}无效的模式和目标组合${NC}"
        show_help
        exit 1
        ;;
esac