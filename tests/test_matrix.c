#include "test.h"

TEST(test){EXPECT(1 == 1)}

TEST(test2) {
    EXPECT(1 == 2)
}

int main(int argc, char **argv) {
    run_tests();

    return 0;
}
