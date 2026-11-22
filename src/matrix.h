#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

#ifdef MATRIX_IMPLEMENTATION
#define ARENA_IMPLEMENTATION
#endif
#include "arena.h"
#include "logs.h"

typedef struct Matrix {
    int rows, cols;
    int rs, cs;
    float *data;
} Matrix;

Matrix matCreate(unsigned int rows, unsigned int cols);
void matDestroy(Matrix *a);
Matrix matDupe(Matrix a);

#define EPS 1e-5
#define ABS(a) (a < 0 ? -a : a)
#define FLOAT_EQ(a, b) (ABS((a) - (b)) < EPS)

// TODO: update
/// Equal by value
bool matEq(Matrix a, Matrix b) {
    if (a.cols != b.cols || a.rows != b.rows || a.cs != b.cs || a.rs != b.rs) {
        return false;
    }

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            int idx = i * a.rs + j * a.rs;
            if (!FLOAT_EQ(a.data[idx], b.data[idx])) {
                return false;
            }
        }
    }
    return true;
}

Matrix matArenaCreate(Arena *arena, unsigned int rows, unsigned int cols);
Matrix matArenaDupe(Arena *arena, Matrix a);
void matZero(Matrix *a);

// Views -------------

Matrix matTransposeView(Matrix a);

// Allocating -----------
Matrix matMul(Arena *arena, Matrix a, Matrix b);
Matrix matScale(Arena *arena, Matrix a, float r);
Matrix matAdd(Arena *arena, Matrix a, Matrix b);
Matrix matSub(Arena *arena, Matrix a, Matrix b);
// [a, b]prod[d, e] = [a*d, b*e]
Matrix matProduct(Arena *arena, Matrix a, Matrix b);
Matrix matDiv(Arena *arena, Matrix a, Matrix b);
Matrix matSqrt(Arena *arena, Matrix a);
Matrix matAddScalar(Arena *arena, Matrix a, float r);

// not sure how to implement it yet
Matrix matSoftMax(Arena *arena, Matrix a);

Matrix matReLu(Arena *arena, Matrix a);
Matrix matReLuDer(Arena *arena, Matrix a);

// Non-allocating -----------
void matMulInto(Matrix *out, Matrix a, Matrix b);
void matScaleInto(Matrix *out, Matrix a, float b);
void matAddInto(Matrix *out, Matrix a, Matrix b);
void matSubInto(Matrix *out, Matrix a, Matrix b);
// [a, b]prod[d, e] = [a*d, b*e]
void matProductInto(Matrix *out, Matrix a, Matrix b);
void matDivInto(Matrix *out, Matrix a, Matrix b);
void matSqrtInto(Matrix *out, Matrix a);
void matAddScalarInto(Matrix *out, Matrix a, float r);

void matSoftMaxInto(Matrix *out, Matrix a);

void matReLuInto(Matrix *out, Matrix a);
void matReLuDerInto(Matrix *out, Matrix a);

void matDebug(Matrix a);

#define MATRIX_IMPLEMENTATION
#ifdef MATRIX_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix matCreateRaw(float *data, unsigned int rows, unsigned int cols) {
    LOG("Created Matrix of size: (%ud, %ud)", rows, cols);
    if (data == NULL) {
        return (Matrix){0};
    }

    return (Matrix){
        .rows = (int)rows,
        .cols = (int)cols,
        .rs = cols,
        .cs = 1,
        .data = data,
    };
}

// zero-initialized
Matrix matCreate(unsigned int rows, unsigned int cols) {
    float *data = (float *)calloc(rows * cols, sizeof(float));
    return matCreateRaw(data, rows, cols);
}

// Add matrix ids
void matDestroy(Matrix *a) {
    if (a->data != NULL)
        free(a->data);
    a->data = NULL;
}

Matrix matArenaCreate(Arena *arena, unsigned int rows, unsigned int cols) {
    float *data = (float *)arenaAlloc(arena, rows * cols * sizeof(float));
    ASSERT(data != NULL, "The allocation failed");

    return matCreateRaw(data, rows, cols);
}

// TODO: maybe do the same abstractin as before
Matrix matDupe(Matrix a) {
    Matrix ret = matCreate(a.rows, a.cols);

    memcpy(ret.data, a.data, a.rows * a.cols * sizeof(float));
    return ret;
}

Matrix matArenaDupe(Arena *arena, Matrix a) {
    Matrix ret = matArenaCreate(arena, a.rows, a.cols);

    memcpy(ret.data, a.data, a.rows * a.cols * sizeof(float));
    return ret;
}

Matrix matMul(Arena *arena, Matrix a, Matrix b) {
    Matrix out = matArenaCreate(arena, a.rows, b.cols);

    matMulInto(&out, a, b);

    return out;
}

// #define USE_OPENBLAS
#ifdef USE_OPENBLAS

#include <cblas.h>

// using cblas
void matMulInto(Matrix *out, Matrix a, Matrix b) {
    ASSERT(a.cols == b.rows, "a.cols and b.rows do not match");
    ASSERT(out->rows == a.rows && out->cols == b.cols,
           "the out size does not match");

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.rows, b.cols,
                a.cols, 1.0f, a.data, a.cols, b.data, b.cols, 1.0, out->data,
                out->cols);
}

#else

// out should be initialized to the right size
void matMulInto(Matrix *out, Matrix a, Matrix b) {
    ASSERT(a.cols == b.rows, "a.cols and b.rows do not match");
    ASSERT(out->rows == a.rows && out->cols == b.cols,
           "the out size does not match");

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            out->data[i * out->cols + j] = 0;
            for (int k = 0; k < a.cols; k++) {
                out->data[i * out->cols + j] +=
                    a.data[i * a.rs + k * a.cs] * b.data[k * b.rs + j * b.cs];
            }
        }
    }
}

#endif // USE_OPENBLAS

Matrix matScale(Arena *arena, Matrix a, float b) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matScaleInto(&out, a, b);

    return out;
}

void matScaleInto(Matrix *out, Matrix a, float b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");

    for (int i = 0; i < a.cols * a.rows; i++) {
        out->data[i] = b * a.data[i];
    }
}

Matrix matAdd(Arena *arena, Matrix a, Matrix b) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matAddInto(&out, a, b);

    return out;
}

void matAddInto(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows,
           "b size does not match a size");

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = a.data[i] + b.data[i];
    }
}

Matrix matSub(Arena *arena, Matrix a, Matrix b) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matSubInto(&out, a, b);

    return out;
}

void matSubInto(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows,
           "b size does not match a size");
    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = a.data[i] - b.data[i];
    }
}

Matrix matTransposeView(Matrix a) {
    Matrix out = (Matrix) {
        .rows = a.cols,
        .cols = a.rows,
        .cs = a.rs,
        .rs = a.cs,
        .data = a.data
    };
    return out;
}

Matrix matProduct(Arena *arena, Matrix a, Matrix b) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matProductInto(&out, a, b);

    return out;
}

void matProductInto(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows,
           "b size does not match a size");

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = a.data[i] * b.data[i];
    }
}

Matrix matDiv(Arena *arena, Matrix a, Matrix b) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matDivInto(&out, a, b);

    return out;
}

void matDivInto(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows,
           "b size does not match a size");

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = a.data[i] / b.data[i];
    }
}

void matZero(Matrix *a) {
    ASSERT(a->data == NULL, "a is unallocated");
    memset(a->data, 0, a->rows * a->cols * sizeof(float));
}

Matrix matSqrt(Arena *arena, Matrix a) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matSqrtInto(&out, a);

    return out;
}

void matSqrtInto(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = sqrtf(a.data[i]);
    }
}

Matrix matAddScalar(Arena *arena, Matrix a, float r) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matAddScalarInto(&out, a, r);

    return out;
}

void matAddScalarInto(Matrix *out, Matrix a, float r) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = a.data[i] + r;
    }
}

Matrix matReLu(Arena *arena, Matrix a) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matReLuInto(&out, a);

    return out;
}

void matReLuInto(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = (a.data[i] > 0) ? a.data[i] : 0;
    }
}

Matrix matReLuDer(Arena *arena, Matrix a) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matReLuDerInto(&out, a);

    return out;
}

void matReLuDerInto(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] = (a.data[i] > 0) ? 1 : 0;
    }
}

Matrix matSoftMax(Arena *arena, Matrix a) {
    Matrix out = matArenaCreate(arena, a.rows, a.cols);

    matSoftMaxInto(&out, a);

    return out;
}

void matSoftMaxInto(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows,
           "out size does not match a size");

    float max_val = a.data[0];
    for (int i = 1; i < a.rows * a.cols; i++) {
        if (a.data[i] > max_val)
            max_val = a.data[i];
    }

    float total = 0.0;
    for (int i = 0; i < a.rows * a.cols; i++) {
        float curr = exp(a.data[i] - max_val);
        out->data[i] = curr;
        total += curr;
    }

    for (int i = 0; i < a.rows * a.cols; i++) {
        out->data[i] /= total;
    }
}

void matDebug(Matrix mat) {
    printf("Matrix (rows: %d, cols: %d, rs: %d, cs: %d):\n", mat.rows, mat.cols,
           mat.rs, mat.cs);
    for (int i = 0; i < mat.rows; i += 1) {
        for (int j = 0; j < mat.cols; j += 1) {
            printf("%5.1f ", mat.data[i * mat.cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

#endif
#endif
