
#include "test.h"


int main(int argc, char **argv) {

    TEST(test) {
        EXPECT(1 == 1)
    }

    TEST(test2) {
        EXPECT(1 == 2)
    }

    return 0;
}
