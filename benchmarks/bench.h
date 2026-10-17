#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <time.h>
#include <stdio.h>

#define BENCH(label, ...)  do {\
    struct timespec start, end;\
    clock_gettime(CLOCK_MONOTONIC, &start);\
    __VA_ARGS__;\
    clock_gettime(CLOCK_MONOTONIC, &end);\
    double elapsed = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/1e9;\
    printf("[BENCH] %-30s %10.4f s\n", label, elapsed);\
  } while (0);\

#endif
