#ifndef INCLUDE_LIB2EASY_LOG_H_
#define INCLUDE_LIB2EASY_LOG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include <stdio.h>

#ifndef pr_fmt
#    define pr_fmt(fmt) fmt
#endif
#define LOG_TIME_CONTENT_SIZE 32
#define ERR " ERROR "
#define INFO " INFO "
#define TODO " TODO "
#define BUG " FATAL "
#define DBG " DEBUG "

#define LOG_DEFAULT_MAX_LINE 2500000
#define LOG_DEFAULT_MAX_FILE_COUNT 2
#define LOG_DEFAULT_MAX_SIZE (1UL << 30) // 1GB

#define LOG_MAX_LINE_LIMIT 20000000UL
#define LOG_MAX_FILE_COUNT_LIMIT 20
#define LOG_MAX_SIZE_LIMIT (1UL << 35) // 32GB

struct log_config {
    unsigned long log_max_line;
    unsigned long log_max_file_count;
    unsigned long log_max_size;
    int self_delete;
};

struct log_instance {
    FILE* utils_log_file;
    pthread_mutex_t mutex;
    int log_lines;
    char log_file_name[1024];
    int dbg_level;
    unsigned long file_idx, last_file_idx;
    unsigned long log_size;
    struct log_config config;
};

extern struct log_instance g_log_instance;

void remove_last_log_file(struct log_instance* i);

void utils_log_format_time(char* buf, int size);

void utils_log_format_user_time(char* buf, int size, unsigned long long dt /*unit millisecond*/);

int utils_log_init__i(struct log_instance* i, char* dir, char* logname, struct log_config* cfg);

int utils_log_init_again__i(struct log_instance* i, char* dir, char* logname, struct log_config* cfg);

int utils_log_init_cfg(
    struct log_config* cfg, unsigned int max_count, unsigned long max_line, unsigned long max_size, int self_del);

int utils_log_inited__i(struct log_instance* i);

int utils_log_has_stale__i(struct log_instance* i);

unsigned long utils_log_obsolete_last_idx(struct log_instance* i);

/* Instance version */
struct log_instance* utils_log_create(char* dir, char* logname, struct log_config* cfg);

void utils_log_destroy(struct log_instance*);

void utils_log_valist(void* log_instance, const char* format, ...) __attribute__((format(printf, 2, 3)));

void utils_log_va(void* log_instance, const char* format, va_list arg);

void utils_log_lock__i(struct log_instance* log);

void utils_log_unlock__i(struct log_instance* log);

void utils_log_flush(void* log_instance);

void utils_log_flush_with_buffer(void* log_instance, const char* buf, size_t buf_size);

void utils_log_output_with_buffer(void* log_instance, const char* buf, size_t buf_size);

int lib2easy_log_debug_level(void* log_instance);

void lib2easy_log_debug_onoff(void* log_instance, int onoff);

void utils_log_checkline__i(struct log_instance* log);

void utils_log_addline__i(struct log_instance* log);

#define utils_log_init(dir, logname, cfg) utils_log_init__i(&g_log_instance, dir, logname, cfg)
#define utils_log_lock() utils_log_lock__i(&g_log_instance)
#define utils_log_unlock() utils_log_unlock__i(&g_log_instance)

#define utils_log_has_stale() utils_log_has_stale__i(&g_log_instance)

#define utils_log_init_again(dir, logname, cfg) utils_log_init_again__i(&g_log_instance, dir, logname, cfg)

#define LOGTODO(fmt, ...) \
    do { \
        char __time_content[LOG_TIME_CONTENT_SIZE]; \
        utils_log_format_time(__time_content, LOG_TIME_CONTENT_SIZE); \
        utils_log_lock(); \
        utils_log_valist(&g_log_instance, "%s" TODO "%s:" pr_fmt(fmt), __time_content, __FUNCTION__, ##__VA_ARGS__); \
        utils_log_flush(&g_log_instance); \
        utils_log_unlock(); \
    } while (0)

#define LOGB(fmt, ...) \
    do { \
        char __time_content[LOG_TIME_CONTENT_SIZE]; \
        utils_log_format_time(__time_content, LOG_TIME_CONTENT_SIZE); \
        utils_log_lock(); \
        utils_log_valist( \
            &g_log_instance, "%s" BUG "%s.%s:" pr_fmt(fmt), __time_content, __FILE__, __FUNCTION__, ##__VA_ARGS__); \
        utils_log_flush(&g_log_instance); \
        utils_log_unlock(); \
    } while (0)

#define LOGE(fmt, ...) \
    do { \
        char __time_content[LOG_TIME_CONTENT_SIZE]; \
        utils_log_format_time(__time_content, LOG_TIME_CONTENT_SIZE); \
        utils_log_lock(); \
        utils_log_valist( \
            &g_log_instance, \
            "%s" ERR "%s.%s.%d:" pr_fmt(fmt), \
            __time_content, \
            __FILE__, \
            __FUNCTION__, \
            __LINE__, \
            ##__VA_ARGS__); \
        utils_log_flush(&g_log_instance); \
        utils_log_unlock(); \
    } while (0)

#define LOGI(fmt, ...) \
    do { \
        char __time_content[LOG_TIME_CONTENT_SIZE]; \
        utils_log_format_time(__time_content, LOG_TIME_CONTENT_SIZE); \
        utils_log_lock(); \
        utils_log_valist(&g_log_instance, "%s" INFO pr_fmt(fmt), __time_content, ##__VA_ARGS__); \
        utils_log_flush(&g_log_instance); \
        utils_log_unlock(); \
    } while (0)

#ifdef DEBUG_LOG
#    define LOGD(__level, fmt, ...) \
        do { \
            char __time_content[LOG_TIME_CONTENT_SIZE]; \
            if (lib2easy_log_debug_level(&g_log_instance) < __level) \
                break; \
            utils_log_format_time(__time_content, LOG_TIME_CONTENT_SIZE); \
            utils_log_lock(); \
            utils_log_valist(&g_log_instance, "%s" DBG pr_fmt(fmt), __time_content, ##__VA_ARGS__); \
            utils_log_flush(&g_log_instance); \
            utils_log_unlock(); \
        } while (0)
#else
#    define LOGD(__level, fmt, ...)
#endif

#define LOG(fmt, ...) \
    do { \
        utils_log_lock(); \
        utils_log_valist(&g_log_instance, pr_fmt(fmt), ##__VA_ARGS__); \
        utils_log_flush(&g_log_instance); \
        utils_log_unlock(); \
    } while (0)

#define LOG_NO_FLUSH(LOG_INSTANCE, fmt, ...) \
    do { \
        char __time_content[LOG_TIME_CONTENT_SIZE]; \
        utils_log_format_time(__time_content, LOG_TIME_CONTENT_SIZE); \
        utils_log_lock__i(LOG_INSTANCE); \
        utils_log_valist(LOG_INSTANCE, "%s" INFO pr_fmt(fmt), __time_content, ##__VA_ARGS__); \
        utils_log_unlock__i(LOG_INSTANCE); \
    } while (0)

#ifdef VERBOSE_LOG
#    define LOGV LOGI
#else
#    define LOGV(format, ...)
#endif
#define LOGW LOGI
#define LOGF LOGI

static __inline void log_dump_hex(void* buf, int size) {
    int i;
    unsigned char* p = (unsigned char*)buf;
    for (i = 0; i < size;) {
        utils_log_valist(&g_log_instance, "%02X ", p[i]);
        if (0 == (++i % 16))
            utils_log_valist(&g_log_instance, "\n");
    }
}

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_LIB2EASY_LOG_H_ */