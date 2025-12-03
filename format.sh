#!/bin/bash

FILE_TYPES=("*.cpp" "*.h" "*.hpp", "*.c")
EXCLUDE_DIRS=("build" "install" "thirdparties" "src/protocol/gen")

CLANG_FORMAT_CONFIG=".clang-format"

if [ ! -f "$CLANG_FORMAT_CONFIG" ]; then
    echo "Err .clang-format $CLANG_FORMAT_CONFIG"
    exit 1
fi

echo "Use: $CLANG_FORMAT_CONFIG"

EXCLUDE_OPTS=()
for dir in "${EXCLUDE_DIRS[@]}"; do
    EXCLUDE_OPTS+=( -path "*/$dir/*" -prune -o )
done

for file_type in "${FILE_TYPES[@]}"; do
    find . "${EXCLUDE_OPTS[@]}" -type f -name "$file_type" -print0 | while IFS= read -r -d '' file; do
        echo "Format: $file"
        clang-format -style=file:"$CLANG_FORMAT_CONFIG" -i "$file"
    done
done

echo "Done!"