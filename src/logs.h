#ifndef LOGS_H
#define LOGS_H

#define ASSERT(cond, fmt, ...) do {\
        if (!(cond)) {\
            LOG(fmt, ##__VA_ARGS__);\
            abort();\
        }\
    } while (0);

#ifdef DEBUG

    #define LOG(fmt, ...) fprintf(stderr, "[LOG][%s:%d]\n" fmt, __FILE__, __LINE__, ##__VA_ARGS__)

#else

    #define LOG(fmt, ...)

#endif

#endif
