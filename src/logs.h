#ifndef LOGS_H
#define LOGS_H

#include <stdio.h>

#define ASSERT(cond, fmt, ...) do {\
        if (!(cond)) {\
            LOG(fmt, ##__VA_ARGS__);\
            abort();\
        }\
    } while (0);

#define LOG(fmt, ...) fprintf(stderr, "[LOG]: [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#endif
