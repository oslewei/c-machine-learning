/// functions are static for now
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "arena.h"
#include "logs.h"
#include "matrix.h"
#include "types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>

typedef struct {
    // layer size x previous layer size
    Matrix weights;
    // size of layer x 1
    Matrix biases;

    // for backpropagation maybe add x z and a
    // x: input, z: weighted sum, a: activation
    // The input to the model would be x of the first layer and the output would be a of the last
    Matrix x, z, a;

    // for  Adam
    Matrix w_m, w_v;
    Matrix b_m, b_v;
} Layer;

typedef struct {
    // the first layer is first hidden layer the layer num_layers-1 is the output
    // currently activation functionn ReLu is assumed between the layers
    // Last layer has no biases
    // The layers have to match, the programm will check this
    // Max 5 layers for now to avoid heap allocation
    Layer layers[5];
    int num_layers;

    // for Adam
    int t;
} NeuralNetwork;

static int nnArenaCreate(NeuralNetwork *network, Arena *arena, u32 in_size, u32 num_layers, ...) {
    network->num_layers = num_layers;
    if(num_layers > 5) return 1;

    va_list args;
    va_start(args, num_layers);

    for (int i = 0; i < (int)num_layers; i++) {
        u32 size = va_arg(args, u32);
        Matrix weights = matArenaCreate(arena, size, in_size);
        if (weights.data == NULL) return 1;
        network->layers[i].weights = weights;

        Matrix w_m = matArenaCreate(arena, size, in_size);
        if (w_m.data == NULL) return 1;
        matZero(&w_m);
        network->layers[i].w_m = w_m;

        Matrix w_v = matArenaDupe(arena, w_m);
        if (w_v.data == NULL) return 1;
        network->layers[i].w_v = w_v;

        in_size = size;

        Matrix biases = matArenaCreate(arena,size, 1);
        if (biases.data == NULL) return 1;
        network->layers[i].biases = biases;

        Matrix b_v = matArenaCreate(arena, size, 1);
        if (b_v.data == NULL) return 1;
        matZero(&b_v);
        network->layers[i].b_v = b_v;

        Matrix b_m = matArenaDupe(arena, b_v);
        if (b_m.data == NULL) return 1;
        network->layers[i].b_m = b_m;
    }

    va_end(args);

    return 0;
}

// Allocates zero-initialized weight and bias matrices for use as a gradient accumulator
static int nnZeroGradients(NeuralNetwork *network, NeuralNetwork *gradients, Arena *arena) {
    gradients->num_layers = network->num_layers;
    for (int i = 0; i < network->num_layers; i++) {
        gradients->layers[i].weights = matArenaCreate(arena, network->layers[i].weights.rows, network->layers[i].weights.cols);
        if (gradients->layers[i].weights.data == NULL) return 1;
        matZero(&gradients->layers[i].weights);
        gradients->layers[i].biases = matArenaCreate(arena, network->layers[i].biases.rows, network->layers[i].biases.cols);
        if (gradients->layers[i].biases.data == NULL) return 1;
        matZero(&gradients->layers[i].biases);
    }
    return 0;
}

static int nnScaleGradients(NeuralNetwork *gradients, float scale) {
    for (int i = 0; i < gradients->num_layers; i++) {
        matScaleInto(&gradients->layers[i].weights, gradients->layers[i].weights, scale);
        matScaleInto(&gradients->layers[i].biases, gradients->layers[i].biases, scale);
    }

    return 0;
}

static int nnAddGradients(NeuralNetwork *gradients, NeuralNetwork *batch_gradients) {
    if (gradients->num_layers != batch_gradients->num_layers) return 1;
    for (int i = 0; i < gradients->num_layers; i++) {
        matAddInto(&gradients->layers[i].weights, gradients->layers[i].weights, batch_gradients->layers[i].weights);
        matAddInto(&gradients->layers[i].biases, gradients->layers[i].biases, batch_gradients->layers[i].biases);
    }

    return 0;
}

// in has to be the size of network->layers[0].weights.cols x 1 and out has to be
// the same size of network->weights[network->num_weights-a] x 1
// network is by reference but is constant
// will update x, z, a for each layer
//
// Currently it has ReLU in each layer and SoftMax at the end as it is specific to MNIST, possibly different activaiton and loss functions after
//
// some sort of
// '
// typedef struct {
//   int (*activation)(Matrix *, Matrix);
//   int (*activationDer)(Matrix *, Matrix);
// } Activation;
// '
static int nnForward(NeuralNetwork *network, Matrix *out, Matrix in, Arena *arena) {
    if (in.rows != network->layers[0].weights.cols || in.cols != 1) {
        fprintf(stderr, "The input is not %u x %d\n", network->layers[0].weights.cols, 1);
        return 1;
    }
    if (out->rows != network->layers[network->num_layers-1].weights.rows || out->cols != 1) {
        fprintf(stderr, "The output is not %u x %d\n", network->layers[network->num_layers-1].weights.rows, 1);
        return 1;
    }

    for (int i = 0; i < network->num_layers; i++) {
        Layer *current = network->layers + i;

        // not great but avoids memory leak, might lead to false frees
        current->x = matArenaDupe(arena, in);

        Matrix temp = matArenaCreate(arena, current->biases.rows, 1);
        if (temp.data == NULL) return 1;
        matMulInto(&temp, current->weights, in);
        matAddInto(&temp, temp, current->biases);

        current->z = matArenaDupe(arena, temp);

        // ReLu ever layer except the last, there is soft max
        if (i < network->num_layers - 1) {
            matReLuInto(&temp, temp);
        } else {
            matSoftMaxInto(&temp, temp);
        }
        current->a = matArenaDupe(arena, temp);

        // only frees it if it is not owned by the caller
        in = temp;
    }

    memcpy(out->data, in.data, out->cols*out->rows*sizeof(float));
    return 0;
}


static int nnOptAdam(NeuralNetwork *network, NeuralNetwork *gradients, Arena *arena, float lr, float beta1, float beta2, float eps) {
    // updating the t
    network->t++;
    for (int i = 0; i < network->num_layers; i++) {
        Layer *curr_layer = network->layers + i;
        // implementing Adam
        //
        Matrix w_g = gradients->layers[i].weights;
        Matrix w_g2 = matArenaDupe(arena, w_g);
        // gettingthe square
        matProductInto(&w_g2, w_g, w_g);

        // updating w_m
        matScaleInto(&curr_layer->w_m, curr_layer->w_m, beta1);
        matScaleInto(&w_g, w_g, (1-beta1));
        matAddInto(&curr_layer->w_m, curr_layer->w_m, w_g);

        // updating w_v
        matScaleInto(&curr_layer->w_v, curr_layer->w_v, beta2);
        matScaleInto(&w_g2, w_g2, (1-beta2));
        matAddInto(&curr_layer->w_v, curr_layer->w_v, w_g2);

        // bias correction
        Matrix w_mc = matArenaDupe(arena, curr_layer->w_m);
        matScaleInto(&w_mc, w_mc, 1.0/(1-powf(beta1, network->t)));
        Matrix w_vc = matArenaDupe(arena, curr_layer->w_v);
        matScaleInto(&w_vc, w_vc, 1.0/(1-powf(beta2, network->t)));

        matSqrtInto(&w_vc, w_vc);
        matAddScalarInto(&w_vc, w_vc, eps);
        matDivInto(&w_mc, w_mc, w_vc);
        matScaleInto(&w_mc, w_mc, lr);
        matSubInto(&curr_layer->weights, curr_layer->weights, w_mc);

        Matrix b_g = gradients->layers[i].biases;
        Matrix b_g2 = matArenaDupe(arena, b_g);
        matProductInto(&b_g2, b_g, b_g);

        // updating b_m
        matScaleInto(&curr_layer->b_m, curr_layer->b_m, beta1);
        matScaleInto(&b_g, b_g, (1-beta1));
        matAddInto(&curr_layer->b_m, curr_layer->b_m, b_g);

        // updating b_v
        matScaleInto(&curr_layer->b_v, curr_layer->b_v, beta2);
        matScaleInto(&b_g2, b_g2, (1-beta2));
        matAddInto(&curr_layer->b_v, curr_layer->b_v, b_g2);

        // biases
        Matrix b_mc = matArenaDupe(arena, curr_layer->b_m);
        matScaleInto(&b_mc, b_mc, 1.0/(1-powf(beta1, network->t)));
        Matrix b_vc = matArenaDupe(arena, curr_layer->b_v);
        matScaleInto(&b_vc, b_vc, 1.0/(1-powf(beta2, network->t)));

        matSqrtInto(&b_vc, b_vc);
        matAddScalarInto(&b_vc, b_vc, eps);
        matDivInto(&b_mc, b_mc, b_vc);
        matScaleInto(&b_mc, b_mc, lr);
        matSubInto(&curr_layer->biases, curr_layer->biases, b_mc);
    }
    return 0;
}

// layers has to have the same length of network->num_layers
static int nnAddGradientsToNetwork(NeuralNetwork *network, NeuralNetwork *gradients, Arena *arena,  float lr) {
    return nnOptAdam(network, gradients, arena, lr, 0.9, 0.999, 1e-8);

}

static int nnBackward(NeuralNetwork *network, NeuralNetwork *gradients, Matrix target, Arena *arena) {
    // output after soft max
    Matrix out = network->layers[network->num_layers-1].a;
    if (target.rows != out.rows || target.cols != out.cols) return 1;

    // focus on the weighs first, the array  of errors
    Matrix *ds = (Matrix *)arenaAlloc(arena, sizeof(Matrix)*network->num_layers);

    // dL is the gradient of C with respect of the pre activation outputs on layer L, L is the last
    Matrix dL = matArenaCreate(arena, target.rows, target.cols);
    matSubInto(&dL, out, target);
    ds[network->num_layers-1] = dL;

    for (int l = network->num_layers-2; l >= 0; l--) {
        Layer *layer = &network->layers[l];
        Layer *next_layer = &network->layers[l+1];

        Matrix wT = matTransposeView(next_layer->weights);

        Matrix templ = matArenaCreate(arena, wT.rows, ds[l+1].cols);
        matMulInto(&templ, wT, ds[l+1]);

        Matrix tempr = matArenaDupe(arena, layer->z);

        matReLuDerInto(&tempr, tempr);

        matProductInto(&templ, templ, tempr);

        ds[l] = templ;
    }

    // Will be directly added
    // NeuralNetwork gradients = {0};
    // if (nnEnsureGradientsSize(network, gradients) != 0) return 1;
    if (gradients->num_layers != network->num_layers) {
        fprintf(stderr, "The gradient does not have the same layers: %du %du\n", network->num_layers, gradients->num_layers);
        return 1;
    };

    for (int i = 0; i < network->num_layers; i++) {
        // copying for now
        gradients->layers[i].biases = ds[i];

        gradients->layers[i].weights = matArenaDupe(arena, network->layers[i].weights);
        // this should be the weights
        // printf("weight size %d x %d\n", network->layers[i].weights.rows, network->layers->weights.cols);
        // printf("%d x %d mul %d x %d\n", network->layers[i].x.rows, network->layers[i].x.cols, ds[i].rows, ds[i].cols);

        // Idk the math yet
        for (int j = 0; j < network->layers[i].weights.rows; j++) {
            for (int k = 0; k < network->layers[i].weights.cols; k++) {
                gradients->layers[i].weights.data[j*gradients->layers[i].weights.cols + k] = network->layers[i].x.data[k]*ds[i].data[j];
            }
        }
    }

    return 0;
}


#endif
