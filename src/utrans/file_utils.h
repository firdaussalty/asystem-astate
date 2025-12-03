#ifndef INCLUDE_LIB2EASY_FILE_UTILS_H
#define INCLUDE_LIB2EASY_FILE_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <alloca.h>
#include <errno.h>
#include <regex.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#ifdef _WIN32
#    include <direct.h>
#else
#    include <sys/stat.h>
#endif

/**
 * @param dir: the path of dir you want to create
 * @param mode: create dir mode, such like "0755"
 * @return if success return 0, else return -1 and set errno
 */
int create_dir(const char* dir, unsigned int mode);

/**
 * @return if legal reutrn 0, else return -1
 */
int check_file_path_legal(const char* dir);

#ifdef __cplusplus
}
#endif
#endif // INCLUDE_LIB2EASY_FILE_UTILS_H