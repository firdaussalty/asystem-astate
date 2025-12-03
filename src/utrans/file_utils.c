#include "file_utils.h"

#if defined(__APPLE__) && defined(__MACH__)
#    include <mach-o/dyld.h>
#endif

int create_dir(const char* dir, unsigned int mode) {
    int dir_len = strlen(dir), ret = 0;
    if (dir_len <= 0) {
        printf("mkdir failed, dir length is %d\n", dir_len);
        ret = -1;
        goto end;
    }
    char* dir_name = (char*)alloca(dir_len + 2);
    strcpy(dir_name, dir);
    if (dir_name[dir_len - 1] != '/') {
        strcat(dir_name, "/");
        ++dir_len;
    }
    int i = dir_name[0] == '/' ? 1 : 0;

    for (; i < dir_len; ++i) {
        if (dir_name[i] == '/') {
            dir_name[i] = 0;
            if (access(dir_name, F_OK) != 0) {
                errno = 0;
#ifndef _WIN32
                if (mkdir(dir_name, mode) != 0) {
#else
                if (_mkdir(dir_name, mode) != 0) {
#endif
                    printf("create dir %s failed. err: %d, reason: %s\n", dir_name, errno, strerror(errno));
                    ret = -1;
                    break;
                }
            }
            dir_name[i] = '/';
        }
    }
end:
    return ret;
}

int check_file_path_legal(const char* dir) {
    int ret = -1;
    regex_t reg;
    const char* pattern = "^[a-zA-Z0-9_/\\.-]+$";
    regcomp(&reg, pattern, REG_EXTENDED);

    regmatch_t pmatch[1];

    if (regexec(&reg, dir, 1, pmatch, 0) == 0) {
        ret = 0;
    }

    regfree(&reg);
    return ret;
}