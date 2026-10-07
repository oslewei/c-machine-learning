#ifndef MATRIX_H
#define MATRIX_H

#include "arena.h"
#include "logs.h"
typedef struct Matrix {
    int rows;
    int cols;
    float *data;
} Matrix;

Matrix matCreate(unsigned int rows, unsigned int cols);
void matDestroy(Matrix *a);
Matrix matDupe(Matrix a);

Matrix matArenaCreate(Arena *arena, unsigned int rows, unsigned int cols);
Matrix matArenaDupe(Arena *arena, Matrix a);

int matMul(Matrix *out, Matrix a, Matrix b);
int matScale(Matrix *out, Matrix a, float r);
int matAdd(Matrix *out, Matrix a, Matrix b);
int matSub(Matrix *out, Matrix a, Matrix b);
int matTranspose(Matrix *out, Matrix a);
// [a, b]prod[d, e] = [a*d, b*e]
int matProduct(Matrix *out, Matrix a, Matrix b);
int matDiv(Matrix *out, Matrix a, Matrix b);
int matZero(Matrix *a);
int matSqrt(Matrix *out, Matrix a);
int matAddScalar(Matrix *out, Matrix a, float r);

// not sure how to implement it yet
int matSoftMax(Matrix *out, Matrix a);

int matReLu(Matrix *out, Matrix a);
int matReLuDer(Matrix *out, Matrix a);

void matDebug(Matrix a);

// #define MATRIX_IMPLEMENTATION
#ifdef MATRIX_IMPLEMENTATION

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// zero-initialized
Matrix matCreate(unsigned int rows, unsigned int cols) {
    LOG("Created Matrix of size: (%ud, %ud)", rows, cols);
    float *data = (float *)calloc(rows*cols, sizeof(float));

    if (data == NULL) {
        return (Matrix) {0};
    }

    return (Matrix){
        .rows = (int)rows,
        .cols = (int)cols,
        .data = data,
    };
}

// Add matrix ids
void matDestroy(Matrix *a) {
    if (a->data != NULL) free(a->data);
    a->data = NULL;
}

Matrix matDupe(Matrix a) {
    Matrix ret = matCreate(a.rows, a.cols);

    memcpy(ret.data, a.data, a.rows*a.cols*sizeof(float));
    return ret;
}

Matrix matArenaCreate(Arena *arena, unsigned int rows, unsigned int cols) {
    float *data = (float *)arenaAlloc(arena, rows*cols*sizeof(float));

    if (data == NULL) {
        return (Matrix) {0};
    }

    return (Matrix){
        .rows = (int)rows,
        .cols = (int)cols,
        .data = data,
    };
}

Matrix matArenaDupe(Arena *arena, Matrix a) {
    Matrix ret = matArenaCreate(arena, a.rows, a.cols);

    memcpy(ret.data, a.data, a.rows*a.cols*sizeof(float));
    return ret;
}

// #define USE_OPENBLAS
#ifdef USE_OPENBLAS

#include <cblas.h>

// using cblas
int matMul(Matrix *out, Matrix a, Matrix b) {
    ASSERT(a.cols == b.rows, "a.cols and b.rows do not match");
    ASSERT(out->rows == a.rows && out->cols == b.cols, "the out size does not match");

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        a.rows,
        b.cols,
        a.cols,
        1.0f,
        a.data, a.cols,
        b.data, b.cols,
        1.0,
        out->data, out->cols
    );
    return 0;
}

#else

// out should be initialized to the right size
int matMul(Matrix *out, Matrix a, Matrix b) {
    ASSERT(a.cols == b.rows, "a.cols and b.rows do not match");
    ASSERT(out->rows == a.rows && out->cols == b.cols, "the out size does not match");

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            out->data[i*out->cols + j] = 0;
            for (int k = 0; k < a.cols; k++) {
                out->data[i*out->cols + j] += a.data[i*a.cols + k]*b.data[k*b.cols + j];
            }
        }
    }

    return 0;
}

#endif //USE_OPENBLAS

int matScale(Matrix *out, Matrix a, float b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");

    for (int i = 0; i < a.cols*a.rows; i++) {
        out->data[i] = b*a.data[i];
    }

    return 0;
}

int matAdd(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows, "b size does not match a size");

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i] + b.data[i];
    }

    return 0;
}

int matSub(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows, "b size does not match a size");
    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i] - b.data[i];
    }
    return 0;
}

int matTranspose(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.rows && out->rows == a.cols, "out size does not match a size");

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            out->data[j*out->cols + i] = a.data[i*a.cols + j];
        }
    }

    return 0;
}

int matProduct(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows, "b size does not match a size");

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i]*b.data[i];
    }

    return 0;
}

int matDiv(Matrix *out, Matrix a, Matrix b) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");
    ASSERT(b.cols == a.cols && b.rows == a.rows, "b size does not match a size");

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i]/b.data[i];
    }

    return 0;
}

int matZero(Matrix *a) {
    if (a->data == NULL) return 1;
    memset(a->data, 0, a->rows*a->cols*sizeof(float));
    return 0;
}

int matSqrt(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = sqrtf(a.data[i]);
    }

    return 0;
}

int matAddScalar(Matrix *out, Matrix a, float r) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = a.data[i] + r;
    }

    return 0;
}

int matReLu(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = (a.data[i] > 0) ? a.data[i] : 0;
    }
    return 0;
}

int matReLuDer(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] = (a.data[i] > 0) ? 1 : 0;
    }

    return 0;
}

int matSoftMax(Matrix *out, Matrix a) {
    ASSERT(out->cols == a.cols && out->rows == a.rows, "out size does not match a size");

    float max_val = a.data[0];
    for (int i = 1; i < a.rows*a.cols; i++) {
        if (a.data[i] > max_val) max_val = a.data[i];
    }

    float total = 0.0;
    for (int i = 0; i < a.rows*a.cols; i++) {
        float curr = exp(a.data[i] - max_val);
        out->data[i] = curr;
        total += curr;
    }

    for (int i = 0; i < a.rows*a.cols; i++) {
        out->data[i] /= total;
    }

    return 0;
}

void matDebug(Matrix mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            printf("%5.1f ", mat.data[i*mat.cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

#endif
#endif
