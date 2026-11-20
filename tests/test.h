#ifndef TEST_H
#define TEST_H

#include <stdbool.h>
#include <stdio.h>

#define COLOR_BOLD "\033[1m"
#define COLOR_RED COLOR_BOLD "\033[31m"
#define COLOR_GREEN COLOR_BOLD "\033[32m"
#define COLOR_RESET "\033[0m"

typedef struct TestContext {
    const char *name;

    bool failed;
    const char *failed_condition;

    bool stop;
} TestContext;

static inline void report_result(TestContext *ctx) {
    if (ctx->failed) {
        printf("[TEST]: [%s] " COLOR_RED "FAILED" COLOR_RESET " with: %s\n",
               ctx->name, ctx->failed_condition);
    } else {
        printf("[TEST]: [%s] " COLOR_GREEN "SUCCEEDED" COLOR_RESET "\n",
               ctx->name);
    }
}

#define TEST(test_name)                                                        \
    for (TestContext ctx = (TestContext){.name = #test_name}; !ctx.stop;       \
         report_result(&ctx), ctx.stop = true)                                 \
        for (bool once = true; once; once = false)

// Expects the condition to be true
#define EXPECT(condition)                                                      \
    do {                                                                       \
        if (!(condition)) {                                                    \
            ctx.failed = true;                                                 \
            ctx.failed_condition = #condition;                                 \
            break;                                                             \
        }                                                                      \
    } while (0);

#endif
