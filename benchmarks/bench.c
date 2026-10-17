#include "bench.h"
#define MATRIX_IMPLEMENTATION
#include "../src/matrix.h"

int main() {
    Matrix matA, matB, matC;
    BENCH("create matrix", { matA = matCreate(1000, 1000); });
    matB = matCreate(1000, 1000);
    matC = matCreate(1000, 1000);

    BENCH("multiply matrix", { matMul(&matC, matA, matB); });

    matDestroy(&matA);
    matDestroy(&matB);
    matDestroy(&matC);
}
