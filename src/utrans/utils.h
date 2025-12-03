#ifndef INCLUDE_LIB2EASY_UTILS_H_
#define INCLUDE_LIB2EASY_UTILS_H_

#include <pthread.h>
#include <setjmp.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <sys/time.h>

static __inline void* lib2easy_malloc_align(size_t size, int alignment) {
    void* p;
    if (alignment == 0)
        alignment = 0x1000;
    if (posix_memalign(&p, alignment, size) != 0)
        return NULL;
    return p;
}

static __inline void lib2easy_free_aligned(void* pointer) {
    free(pointer);
}

static __inline unsigned long align_to(unsigned long addr, unsigned long align) {
    align--;
    if (addr & align) {
        addr |= align;
        addr++;
    }
    return addr;
}

static __inline unsigned int align_ui_to(unsigned int value, unsigned int align) {
    align--;
    if (value & align) {
        value |= align;
        value++;
    }
    return value;
}

static __inline void* align_address_to(void* address, unsigned long align) {
    unsigned long long addr = (unsigned long long)address;
    align--;
    if (addr & align) {
        addr |= align;
        addr++;
    }
    return (void*)addr;
}

static __inline int
time_interleave(unsigned long long s1, unsigned long long e1, unsigned long long s2, unsigned long long e2) {
    if (s1 <= s2) {
        if (e1 >= s2)
            return 1;
    } else {
        if (e2 >= s1)
            return 1;
    }

    return 0;
}

void sleep_second(int X);

static inline int64_t lib2easy_get_time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}

static inline int64_t lib2easy_get_time_in_us() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000L;
}

static inline int64_t lib2easy_get_time_in_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline void lib2easy_set_timezone(const char* tz) {
    if (setenv("TZ", tz, 1) != 0) {
        char errmsg[1024];
        snprintf(errmsg, 1024, "set timezone failed, errno=%d, errmsg=%s\n", errno, strerror(errno));
        perror(errmsg);
        exit(-1);
    }
    tzset();
}

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/**
 * align size to 2^n where 2^(n-1) < size <= 2^n
 * @note time-consuming but returns the index
 * @return index of n 
 */
static __inline int align_power_of_2_with_index(size_t* size) {
    if (*size != 1) {
        int index = (64 - __builtin_clzll((*size) - 1));
        (*size) = (size_t)1 << index;
        return index;
    } else {
        return 0;
    }
}

/**
 * align size to 2^n where 2^(n-1) < size <= 2^n 
 * @note index is not returned but is less time-consuming
 * @return 2^n
 */
static __inline size_t align_pow_of_two_fast(size_t size) {
    return size != 1 ? ((size_t)1 << (64 - __builtin_clzll(size - 1))) : (1);
}

#ifdef __cplusplus
}
#endif

#endif // INCLUDE_LIB2EASY_UTILS_H_