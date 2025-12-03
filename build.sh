#!/bin/bash
set -euo pipefail

BUILD_MODE="develop"
BUILD_DIR="build"
JOBS=8

show_help() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  develop     Development mode (default) - compile with tests and debug info"
    echo "  release     Release mode - optimized compilation, no tests"
    echo ""
    echo "Options:"
    echo "  clean       Clean build directory"
    echo "  test        Run all tests"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build with default develop mode"
    echo "  $0 develop            # Explicitly build with develop mode"
    echo "  $0 release            # Build with release mode"
    echo "  $0 clean              # Clean build directory"
    echo "  $0 test               # Run tests"
    echo "  $0 develop clean      # Clean then build with develop mode"
}

clean_build() {
    echo "🧹 Cleaning build directory $BUILD_DIR..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        echo "✅ Build directory cleaned"
    else
        echo "ℹ️  Build directory does not exist, nothing to clean"
    fi
}

run_tests() {
    if [ ! -d "$BUILD_DIR" ]; then
        echo "❌ Build directory does not exist, please build the project first"
        exit 1
    fi

    echo "🧪 Running tests..."
    local current_dir="$(pwd)"
    cd "$BUILD_DIR"

    local junit_output="${current_dir}/test_results.xml"

    if command -v ctest > /dev/null; then
        echo "Using ctest to run tests..."
        if ctest --output-on-failure -L "astate_test" --output-junit "$junit_output"; then
            echo "✅ Tests passed"
            echo "   Results saved to: $junit_output"
        else
            echo "❌ Tests failed"
            cd "$current_dir"
            exit 1
        fi
    else
        echo "ctest not found, running test executables directly..."
        local test_failed=false

        find . -name "*_test" -type f -executable | while read test_exec; do
            echo "Running test: $(basename "$test_exec")"
            if ! "$test_exec"; then
                echo "❌ Test failed: $test_exec"
                test_failed=true
            fi
        done

        if [ "$test_failed" = true ]; then
            cd "$current_dir"
            exit 1
        fi
    fi

    cd "$current_dir"
}

# Develop mode build
build_develop() {
    echo "🔨 Building in develop mode..."
    echo "  - Enable test compilation (project code only)"
    echo "  - Enable compile commands export"
    echo "  - Enable debug information"
    echo "  - Disable third-party library tests"

    cmake \
        -DCMAKE_BUILD_TYPE=Debug \
        -DASTATE_ENABLE_TESTS=ON \
        -DBUILD_TESTING=ON \
        -S . -B "$BUILD_DIR"

    if [ $? -ne 0 ]; then
        echo "❌ CMake configuration failed"
        exit 1
    fi

    cmake --build "$BUILD_DIR" -j $JOBS

    if [ $? -eq 0 ]; then
        echo "✅ Develop mode build successful"
    else
        echo "❌ Build failed"
        exit 1
    fi
}

build_release() {
    echo "🚀 Building in release mode..."
    echo "  - Optimized compilation"
    echo "  - No test compilation"
    echo "  - No compile commands export"

    cmake \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DASTATE_ENABLE_TESTS=OFF \
        -DBUILD_TESTING=OFF \
        -S . -B "$BUILD_DIR"

    if [ $? -ne 0 ]; then
        echo "❌ CMake configuration failed"
        exit 1
    fi

    cmake --build "$BUILD_DIR" -j $JOBS

    if [ $? -eq 0 ]; then
        echo "✅ Release mode build successful"
    else
        echo "❌ Build failed"
        exit 1
    fi
}

main() {
    local should_clean=false
    local should_test=false
    local should_build=true

    while [[ $# -gt 0 ]]; do
        case $1 in
            develop)
                BUILD_MODE="develop"
                shift
                ;;
            release)
                BUILD_MODE="release"
                shift
                ;;
            clean)
                should_clean=true
                shift
                ;;
            test)
                should_test=true
                should_build=false
                shift
                ;;
            help|--help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "❌ Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [ "$should_clean" = true ] && [ "$should_test" = true ]; then
        echo "⚠️  Cannot run tests after cleaning. Please build first."
        echo "   Try: $0 $BUILD_MODE && $0 test"
        exit 1
    fi

    if [ "$should_clean" = true ]; then
        clean_build
    fi

    if [ "$should_test" = true ]; then
        run_tests
        exit 0
    fi

    if [ "$should_build" = true ]; then
        echo "📦 Starting project build (mode: $BUILD_MODE)..."

        case $BUILD_MODE in
            develop)
                build_develop
                ;;
            release)
                build_release
                ;;
            *)
                echo "❌ Unknown build mode: $BUILD_MODE"
                exit 1
                ;;
        esac

        echo "🎉 Build completed!"
        echo ""
        echo "💡 Tips:"
        echo "  - Run tests: $0 test"
        echo "  - Clean and rebuild: $0 clean $BUILD_MODE"
    fi
}

# Check if we're in the correct directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ CMakeLists.txt not found, please run this script from the project root directory"
    exit 1
fi

# Run main function
main "$@"