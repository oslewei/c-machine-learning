#ifndef TEST_H
#define TEST_H

#include <stdbool.h>
#include <stdio.h>

#define COLOR_BOLD "\033[1m"
#define COLOR_RED COLOR_BOLD "\033[31m"
#define COLOR_GREEN COLOR_BOLD "\033[32m"
#define COLOR_RESET "\033[0m"

typedef struct TestContext {
    bool failed;
    const char *failed_condition;

    bool stop;
} TestContext;

typedef void (*TestFunction)(TestContext *);

typedef struct TestFunctionEntry {
    const char *name;
    TestFunction function;
} TestFunctionEntry;

extern TestFunctionEntry __start_tests[];
extern TestFunctionEntry __stop_tests[];

static inline void report_result(TestContext *ctx, TestFunctionEntry *entry) {
    if (ctx->failed) {
        printf("[TEST]: [%s] " COLOR_RED "FAILED" COLOR_RESET " with: %s\n",
               entry->name, ctx->failed_condition);
    } else {
        printf("[TEST]: [%s] " COLOR_GREEN "SUCCEEDED" COLOR_RESET "\n",
               entry->name);
    }
}

static inline void run_tests() {
    for (TestFunctionEntry *test = __start_tests; test < __stop_tests; test++) {
        TestContext ctx = {0};
        (*test->function)(&ctx);
        report_result(&ctx, test);
    }
}

#define TEST(test_name)                                                        \
    void test_name(TestContext *ctx);                                          \
    __attribute__((used, section("tests"))) static TestFunctionEntry           \
        test_name##_registration = {.function = test_name,                     \
                                    .name = #test_name};                       \
    void test_name(TestContext *ctx)

// #define TEST(test_name) \
//     for (TestContext ctx = (TestContext){.name = #test_name}; !ctx.stop; \
//          report_result(&ctx), ctx.stop = true) \
//         for (bool once = true; once; once = false)

// Expects the condition to be true
#define EXPECT(condition)                                                      \
    do {                                                                       \
        if (!(condition)) {                                                    \
            ctx->failed = true;                                                \
            ctx->failed_condition = #condition;                                \
            return;                                                            \
        }                                                                      \
    } while (0);

#endif
