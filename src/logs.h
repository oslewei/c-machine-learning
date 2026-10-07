#ifndef LOGS_H
#define LOGS_H

#include <stdlib.h>
#include <stdio.h>

#ifndef DEBUG

    #define LOG(fmt, ...) fprintf(stderr, "[LOG][%s:%d] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)
    #define ASSERT(cond, fmt, ...) do {\
            if (!(cond)) {\
                LOG(fmt, ##__VA_ARGS__);\
                abort();\
            }\
        } while (0)

#else

    #define LOG(fmt, ...)
    #define ASSERT(cond, fmt, ...)

#endif

#endif
