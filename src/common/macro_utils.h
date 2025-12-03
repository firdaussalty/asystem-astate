#pragma once


namespace astate {

#define DISALLOW_COPY(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete

#define DISALLOW_MOVE(ClassName) \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete

#define DISALLOW_COPY_AND_MOVE(ClassName) \
    DISALLOW_COPY(ClassName); \
    DISALLOW_MOVE(ClassName)

} // namespace astate