#define _CRT_SECURE_NO_DEPRECATE 1

#include "utils.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lib2easy_yield() {
#ifdef _MSC_VER
    Sleep(0);
#else
    sched_yield();
#endif
}

int split_ipport(char* ipport, char* retIp, int* retPort) {
    int idx;
    for (idx = 0; ipport[idx] != '\0'; idx++) {
        if (ipport[idx] == ':') {
            ipport[idx] = '\0';
            strcpy(retIp, ipport);
            ipport[idx] = ':';
            *retPort = atoi(&ipport[idx + 1]);
            return 0;
        }
    }
    return 1;
}