/// Does not test everything, mainly tests the tranpso se and multiply for now

#include "test.h"
#include <math.h>
#define MATRIX_IMPLEMENTATION
#include "matrix.h"

TEST(matrix_multiply) {
    Matrix a = (Matrix){
        .rows = 4,
        .cols = 3,
        .rs = 3,
        .cs = 1,
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
        .rs = 4,
        .cs = 1,
        .data = (float[]){
            12.0, 11.0, 10.0, 9.0,
            8.0, 7.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0
        },
    };
    Matrix out = (Matrix){
        .cols = 4,
        .rows = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[16]){},
    };

    Matrix expected = (Matrix){
        .rows = 4,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[16]){
            40.0, 34.0, 28.0, 22.0,
            112.0, 97.0, 82.0, 67.0,
            184.0, 160.0, 136.0, 112.0,
            256.0, 223.0, 190.0, 157.0
        },
    };

    matMulInto(&out, a, b);

    EXPECT(matEq(out, expected))
}

TEST(matrix_multiply_identity) {
    Matrix id = (Matrix){
        .rows = 3,
        .cols = 3,
        .rs = 3,
        .cs = 1,
        .data = (float[9]){
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        },
    };
    Matrix out = (Matrix){
        .cols = 3,
        .rows = 3,
        .rs = 3,
        .cs = 1,
        .data = (float[9]){},
    };

    matMulInto(&out, id, id);

    for (int i = 0; i < 9; i++) {
        EXPECT(FLOAT_EQ(out.data[i], id.data[i]))
    }
}

TEST(matrix_trasnspose) {
    Matrix a = (Matrix){
        .rows = 4,
        .cols = 3,
        .rs = 3,
        .cs = 1,
        .data = (float[]){
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        },
    };

    Matrix expected = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 1,
        .cs = 3,
        .data = (float[]){
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        },
    };

    Matrix out = matTransposeView(a);

    EXPECT(matEq(expected, out))
}

TEST(matrix_scale) {
    Matrix a = (Matrix){
        .rows = 4,
        .cols = 3,
        .rs = 3,
        .cs = 1,
        .data = (float[]){
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        },
    };

    Matrix expected = (Matrix){
        .rows = 4,
        .cols = 3,
        .rs = 3,
        .cs = 1,
        .data = (float[]){
            2.0, 4.0, 6.0,
            8.0, 10.0, 12.0,
            14.0, 16.0, 18.0,
            20.0, 22.0, 24.0
        },
    };

    Matrix out = (Matrix) {
        .rows = 4,
        .cols = 3,
        .rs = 3,
        .cs = 1,
        .data = (float[12]){
        },
    };

    matScaleInto(&out, a, 2.0);

    EXPECT(matEq(expected, out))
}

TEST(matrix_add) {
    Matrix a = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[]){
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        },
    };
    Matrix b = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[]){
            12.0, 11.0, 10.0, 9.0,
            8.0, 7.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0
        },
    };
    Matrix out = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[16]){},
    };

    Matrix expected = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[]){
            13.0, 13.0, 13.0, 13.0,
            13.0, 13.0, 13.0, 13.0,
            13.0, 13.0, 13.0, 13.0,
        },
    };

    matAddInto(&out, a, b);

    EXPECT(matEq(out, expected));
}

TEST(matrix_sub) {
    Matrix a = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[]){
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        },
    };
    Matrix b = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[]){
            12.0, 11.0, 10.0, 9.0,
            8.0, 7.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0
        },
    };
    Matrix out = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[16]){},
    };

    Matrix expected = (Matrix){
        .rows = 3,
        .cols = 4,
        .rs = 4,
        .cs = 1,
        .data = (float[]){
            -11.0, -9.0, -7.0, -5.0,
            -3.0, -1.0, 1.0, 3.0,
            5.0, 7.0, 9.0, 11.0,
        },
    };

    matSubInto(&out, a, b);

    EXPECT(matEq(out, expected));
}

int main(int argc, char **argv) {
    run_tests();

    return 0;
}
