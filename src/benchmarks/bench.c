#include "benchmark.h"
#define MATRIX_IMPLEMENTATION
#include "../matrix.h"

int main() {
    BENCH("create matrix", {
        Matrix mat = matCreate(1000, 1000);
        matZero(&mat);
        matDestroy(&mat);
    });
}
