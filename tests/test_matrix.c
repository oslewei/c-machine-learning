#include "test.h"
#define MATRIX_IMPLEMENTATION
#include "matrix.h"

TEST(matrix_multiply) {
    Matrix a = (Matrix){
        .rows = 4,
        .cols = 3,
        .data = (float[]){
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        },
    };
    Matrix b = (Matrix){
        .rows = 3,
        .cols = 4,
        .data = (float[]){
            12.0, 11.0, 10.0, 9.0,
            8.0, 7.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0
        },
    };
    Matrix out = (Matrix){
        .cols = 4,
        .rows = 4,
        .data = (float[16]){},
    };

    float expected[16] = {
        40.0, 34.0, 28.0, 22.0,
        112.0, 97.0, 82.0, 67.0,
        184.0, 160.0, 136.0, 112.0,
        256.0, 223.0, 190.0, 157.0
    };

    matMul(&out, a, b);

    for (int j = 0; j < out.rows; j++) {
        for (int i = 0; i < out.cols; i++) {
            EXPECT(FLOAT_EQ(out.data[j*out.cols + i], expected[j*out.cols + i]))
        }
    }
}

int main(int argc, char **argv) {
    run_tests();

    return 0;
}
