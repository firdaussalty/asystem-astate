#define _CRT_SECURE_NO_DEPRECATE 1

#ifdef _MSC_VER
#    include <Windows.h>
#endif

#include <dirent.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/types.h>

#include "file_utils.h"
#include "log.h"
#include "utils.h"

struct log_instance g_log_instance;

/************************************************************************/
/*                                                                      */
/************************************************************************/

static void open_log_file(struct log_instance* i) {
    i->utils_log_file = fopen(i->log_file_name, "wb+");
    if (i->utils_log_file == NULL)
        i->utils_log_file = stdout;
}

static void close_log_file(struct log_instance* i) {
    if (i->utils_log_file && i->utils_log_file != stdout)
        fclose(i->utils_log_file);
    i->utils_log_file = NULL;
}

static void remove_log_file(struct log_instance* i, unsigned long idx) {
    char* name;
    unsigned int len;

    len = strlen(i->log_file_name) + 20 /* reserved for suffix */ + 2 /* end char */;
    name = alloca(len);
    snprintf(name, len, "%s.%lu", i->log_file_name, idx);
    if (remove(name) < 0) {
        printf("Remove log file %s failed. err %d, errstr %s\n", name, errno, strerror(errno));
    }
}

static void move_log_file(struct log_instance* i, unsigned long idx) {
    char* name;
    unsigned int len;

    // No log file, just return
    if (strnlen(i->log_file_name, sizeof(i->log_file_name)) == 0) {
        return;
    }

    len = strlen(i->log_file_name) + 20 /* reserved for suffix */ + 2 /* end char */;
    name = alloca(len);

    snprintf(name, len, "%s.%lu", i->log_file_name, idx);

    if (rename(i->log_file_name, name) < 0) {
        printf("Rename log file %s to %s failed. err %d, errstr %s\n", i->log_file_name, name, errno, strerror(errno));
    }

    if (i->last_file_idx == 0)
        i->last_file_idx = idx;

    if (idx - i->last_file_idx >= i->config.log_max_file_count && i->config.self_delete) {
        remove_last_log_file(i);
    }
}

static int
calc_init_log_idx(const char* dir_name, const char* name, unsigned long* newest_idx, unsigned long* last_idx) {
    char str[1024], *str_ptr, *end_ptr = NULL;
    char suf_name[64];
    DIR* log_dir;
    struct dirent* log_dire;
    struct stat stat_buf;
    unsigned long idx;
    int ret = 0, exists = 0;

    *newest_idx = 0;
    *last_idx = 0;

    log_dir = opendir(dir_name);
    if (!log_dir) {
        printf("Open log dir %s failed. err: %d, reason: %s\n", dir_name, errno, strerror(errno));
        ret = -errno;
        goto end;
    }

    snprintf(suf_name, sizeof(suf_name), "%s.log", name);

    while (NULL != (log_dire = readdir(log_dir))) {
        snprintf(str, sizeof(str), "%s/%s", dir_name, log_dire->d_name);
        if (lstat(str, &stat_buf) < 0) {
            printf(
                "[%s:%d] Stat file %s failed. errno: %d, err: %s\n",
                __FUNCTION__,
                __LINE__,
                str,
                errno,
                strerror(errno));
            continue;
        }
        if (S_ISDIR(stat_buf.st_mode)) {
            continue;
        }

        if (strstr(log_dire->d_name, suf_name) == log_dire->d_name) {
            str_ptr = log_dire->d_name + strlen(suf_name);
            if ('\0' == *str_ptr) {
                exists = 1;
                continue;
            }
            if (*str_ptr != '.') {
                continue;
            }

            str_ptr++;
            idx = strtoul(str_ptr, &end_ptr, 10);
            if (NULL == end_ptr || *end_ptr != '\0') {
                continue;
            }

            if (idx > *newest_idx)
                *newest_idx = idx;

            if (idx < *last_idx || *last_idx == 0)
                *last_idx = idx;
        }
    }

    if (exists) {
        snprintf(str, sizeof(str), "%s/%s", dir_name, suf_name);
        // reuse `str_ptr` here
        str_ptr = alloca(sizeof(str));
        snprintf(str_ptr, sizeof(str), "%s/%s.%lu", dir_name, suf_name, *newest_idx + 1);
        if (rename(str, str_ptr) < 0) {
            printf("Rename log file %s to %s failed. err: %d, reason: %s\n", str, str_ptr, errno, strerror(errno));
        } else {
            (*newest_idx)++;
        }
    }

    closedir(log_dir);
end:
    return ret;
}

void remove_last_log_file(struct log_instance* i) {
    unsigned long idx = utils_log_obsolete_last_idx(i);
    remove_log_file(i, idx);
}

int utils_log_init__i(struct log_instance* i, char* dir_name, char* logname, struct log_config* cfg) {
    unsigned long new_idx, last_idx;
    int ret = 0;

    if (utils_log_inited__i(i)) {
        printf("Log already inited. new prefix is %s, old filename is %s.\n", logname, i->log_file_name);
        goto end;
    }
    memset(i, 0, sizeof(*i));
    pthread_mutex_init(&i->mutex, NULL);

    printf("Creating log file for dir %s, logname %s.\n", dir_name, logname);

    /* TODO: Check name length */
    ret = create_dir(dir_name, 0755);
    if (ret != 0) {
        printf("mkdir failed, path=%s, err=%d, reason=%s\n", dir_name, errno, strerror(errno));
        ret = -errno;
        goto err;
    }

    if ((ret = calc_init_log_idx(dir_name, logname, &new_idx, &last_idx)) < 0) {
        printf("Calc previous log file failed.\n");
        goto err;
    }

    snprintf(i->log_file_name, sizeof(i->log_file_name), "%s/%s.log", dir_name, logname);

    if (cfg) {
        memcpy(&i->config, cfg, sizeof(*cfg));
    } else {
        i->config.log_max_file_count = LOG_DEFAULT_MAX_FILE_COUNT;
        i->config.log_max_line = LOG_DEFAULT_MAX_LINE;
        i->config.log_max_size = LOG_DEFAULT_MAX_SIZE;
        i->config.self_delete = 1;
    }

    LOGI("[LOG CONF] log_file_name=%s, max_line=%lu\n", i->log_file_name, i->config.log_max_line);
    LOGI("[LOG CONF] log_file_name=%s, max_file_count=%lu\n", i->log_file_name, i->config.log_max_file_count);
    LOGI("[LOG CONF] log_file_name=%s, max_size=%lu\n", i->log_file_name, i->config.log_max_size);
    LOGI("[LOG CONF] log_file_name=%s, self_delete=%d\n", i->log_file_name, i->config.self_delete);

    i->log_lines = 0;
    i->file_idx = new_idx;
    i->last_file_idx = last_idx;
    i->log_size = 0;

    open_log_file(i);

    // log 初始化时清掉过期的 log 文件
    while (utils_log_has_stale__i(i)) {
        remove_last_log_file(i);
    }

end:
    return ret;

err:
    close_log_file(i);
    pthread_mutex_destroy(&i->mutex);
    goto end;
}

int utils_log_init_again__i(struct log_instance* i, char* dir, char* logname, struct log_config* cfg) {
    if (utils_log_inited__i(i)) {
        // clear previous content.
        close_log_file(i);
        pthread_mutex_destroy(&i->mutex);
    }

    return utils_log_init__i(i, dir, logname, cfg);
}

int utils_log_init_cfg(
    struct log_config* cfg, unsigned int max_count, unsigned long max_line, unsigned long max_size, int self_del) {
    if (max_count > LOG_MAX_FILE_COUNT_LIMIT) {
        LOGE("[LOG CONF] Arg is invaild. max_count=%u, max_count_limit=%u\n", max_count, LOG_MAX_FILE_COUNT_LIMIT);
        return -1;
    }
    if (max_line > LOG_MAX_LINE_LIMIT) {
        LOGE("[LOG CONF] Arg is invaild. max_line=%lu, max_line_limit=%lu\n", max_line, LOG_MAX_LINE_LIMIT);
        return -1;
    }
    if (max_size > LOG_MAX_SIZE_LIMIT) {
        LOGE("[LOG CONF] Arg is invaild. max_size=%lu, max_size_limit=%lu\n", max_size, LOG_MAX_SIZE_LIMIT);
        return -1;
    }

    cfg->log_max_line = max_line;
    cfg->log_max_file_count = max_count;
    cfg->log_max_size = max_size;
    cfg->self_delete = (!self_del) ? 0 : 1;

    return 0;
}

int utils_log_inited__i(struct log_instance* i) {
    return i && i->utils_log_file ? 1 : 0;
}

int utils_log_has_stale__i(struct log_instance* i) {
    return i->file_idx - i->last_file_idx >= i->config.log_max_file_count;
}

unsigned long utils_log_obsolete_last_idx(struct log_instance* i) {
    return i->last_file_idx++;
}

struct log_instance* utils_log_create(char* dir, char* logname, struct log_config* cfg) {
    struct log_instance* log;
    int ret;

    if (!(log = malloc(sizeof(*log))))
        return NULL;
    memset(log, 0, sizeof(*log));

    ret = utils_log_init__i(log, dir, logname, cfg);
    if (ret < 0) {
        printf("Create log file %s/%s failed. ret = %d\n", dir, logname, ret);
        free(log);
        log = NULL;
    }

    return log;
}

void utils_log_destroy(struct log_instance* i) {
    if (i) {
        close_log_file(i);
        pthread_mutex_destroy(&i->mutex);
        free(i);
    }
}

void utils_log_addline__i(struct log_instance* i) {
    i->log_lines++;
}

void utils_log_checkline__i(struct log_instance* log) {
    /* Check line/size is beyond limit? we have to split a new file */
    if (log->log_lines > log->config.log_max_line || log->log_size >= log->config.log_max_size) {
        close_log_file(log);
        move_log_file(log, ++log->file_idx);

        open_log_file(log);
        log->log_lines = 0;
        log->log_size = 0;
    }
}

void utils_log_lock__i(struct log_instance* i) {
    if (utils_log_inited__i(i)) {
        pthread_mutex_lock(&i->mutex);
        utils_log_addline__i(i);
    }
}

void utils_log_unlock__i(struct log_instance* log) {
    if (utils_log_inited__i(log)) {
        utils_log_checkline__i(log);
        pthread_mutex_unlock(&log->mutex);
    }
}

void utils_log_valist(void* log_instance, const char* format, ...) {
    va_list va;
    struct log_instance* log;
    log = log_instance;
    va_start(va, format);
    if (utils_log_inited__i(log)) {
        log->log_size += vfprintf(log->utils_log_file, format, va);
    } else {
        vprintf(format, va);
    }
    va_end(va);
}

void utils_log_va(void* log_instance, const char* format, va_list arg) {
    struct log_instance* log;
    log = log_instance;
    if (utils_log_inited__i(log)) {
        log->log_size += vfprintf(log->utils_log_file, format, arg);
    } else {
        vprintf(format, arg);
    }
}

void utils_log_flush(void* log_instance) {
    struct log_instance* log;
    log = log_instance;
    if (utils_log_inited__i(log)) {
        fflush(log->utils_log_file);
    } else {
        fflush(stdout);
    }
}

void utils_log_flush_with_buffer(void* log_instance, const char* buf, size_t buf_size) {
    struct log_instance* log;
    FILE* f;

    log = log_instance;
    if (utils_log_inited__i(log)) {
        f = log->utils_log_file;
        log->log_size += fwrite(buf, buf_size, 1, f);
    } else {
        f = stdout;
        fwrite(buf, buf_size, 1, f);
    }
    fflush(f);
}

void utils_log_output_with_buffer(void* log_instance, const char* buf, size_t buf_size) {
    struct log_instance* log;
    FILE* f;
    log = log_instance;
    if (utils_log_inited__i(log)) {
        f = log->utils_log_file;
        log->log_size += fwrite(buf, buf_size, 1, f);
    } else {
        f = stdout;
        fwrite(buf, buf_size, 1, f);
    }
}

int lib2easy_log_debug_level(void* log_instance) {
    struct log_instance* log;
    log = log_instance;

    return log->dbg_level;
}

void lib2easy_log_debug_onoff(void* log_instance, int onoff) {
    struct log_instance* log;
    log = log_instance;

    log->dbg_level = onoff;
}

/************************************************************************/
/*                                                                      */
/************************************************************************/

struct timeval__ {
    long tv_sec; /* seconds */
    long tv_usec; /* and microseconds */
};

void utils_log_format_time(char* buf, int size) {
    time_t timep;
    struct tm p;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    time(&timep);

    localtime_r(&timep, &p);
    snprintf(
        buf,
        size,
        "%04d-%02d-%02d %02d:%02d:%02d,%03ld",
        p.tm_year + 1900,
        p.tm_mon + 1,
        p.tm_mday,
        p.tm_hour,
        p.tm_min,
        p.tm_sec,
        tv.tv_usec / 1000L);
}

void utils_log_format_user_time(char* buf, int size, unsigned long long dt) {
    time_t timep;
    struct tm p;
    timep = (time_t)(dt / 1000) /* seconds */;
    localtime_r(&timep, &p);
    snprintf(
        buf,
        size,
        "%04d-%02d-%02d %02d:%02d:%02d,%03llu",
        p.tm_year + 1900,
        p.tm_mon + 1,
        p.tm_mday,
        p.tm_hour,
        p.tm_min,
        p.tm_sec,
        dt % 1000);
}