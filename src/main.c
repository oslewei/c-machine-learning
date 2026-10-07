#include "arena.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define MATRIX_IMPLEMENTATION
#include "matrix.h"
#include "types.h"
#include "neuralnetwork.h"

#define IMAGE_SIZE 28*28


// row_major
typedef struct {
    u8 data[IMAGE_SIZE];
} Image;

typedef struct {
    Image image;
} LabelledImage;

// easier for now
typedef struct {
    Image *images;
    u8 *labels;
    u32 num_images;
} LabelledImages;

u32 swap32(u32 x) {
    return ((x >> 24) & 0x000000FF) |
           ((x >> 8)  & 0x0000FF00) |
           ((x << 8)  & 0x00FF0000) |
           ((x << 24) & 0xFF000000);
}

// the length of the image and the label file should be the same
// You own the array
// returns sucess == 1
int decodeImages(LabelledImages *labelled_images, const char *images, const char *labels) {
    // read images
    FILE *image_file = fopen(images, "r");
    
    if (fgetc(image_file) != 0) return 0;
    if (fgetc(image_file) != 0) return 0;
    
    u8 type = fgetc(image_file);
    if (type != 0x08) return 0;
    u8 num_dims = fgetc(image_file);
    if (num_dims != 0x03) return 0;
    // printf("type: %d, dimensions: %d\n", type, num_dims);
    
    u32 dimensions[3] = {0};
    fread(dimensions, sizeof(int), 3, image_file);
    for (int i =0; i < 3; i++) {
        dimensions[i] = swap32(dimensions[i]);
    }
    
    // printf("dimensions: (%d, %d, %d)\n", dimensions[0], dimensions[1], dimensions[2]);
    
    labelled_images->images = malloc(dimensions[0]*dimensions[1]*dimensions[2]);
    if (labelled_images->images == NULL) return 0;
    fread(labelled_images->images, 1, dimensions[0]*dimensions[1]*dimensions[2], image_file);
    fclose(image_file);
    
    // read images
    FILE *label_file = fopen(labels, "r");
    
    if (fgetc(label_file) != 0) return 0;
    if (fgetc(label_file) != 0) return 0;
    
    type = fgetc(label_file);
    if (type != 0x08) return 0;
    num_dims = fgetc(label_file);
    if (num_dims != 0x01) return 0;
    // printf("Labels type: %d, dimensions: %d\n", type, num_dims);
    
    u32 label_dimension;
    fread(&label_dimension, sizeof(int), 1, label_file);
    label_dimension = swap32(label_dimension);
    
    // printf("label_dimension: %d\n", label_dimension);
    
    labelled_images->labels = malloc(label_dimension);
    fread(labelled_images->labels, 1, label_dimension, label_file);
    fclose(label_file);
    
    labelled_images->num_images = label_dimension;
    
    return 1;
}

void freeLabelledImages(LabelledImages images) {
    free(images.images);
    free(images.labels);
}
 
void showImage(Image *image) {
    const char options[9] = " .:+=x$#";
    for (int i = 0; i < 28; i++) {
        printf("=");
    }
    printf("\n");
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            u8 idx = (image->data[28*i + j])/32;
            printf("%c", options[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

// -------------------------------------------------------- Machine Learnign

void shuffleIndeces(u32 *indeces, u32 num_indeces) {
    for (int i = 0; i < num_indeces; i++) {
        int j = rand() % num_indeces;
        u32 temp = indeces[i];
        indeces[i] = indeces[j];
        indeces[j] = temp;
    }
}

void plotAccuracies(float *accuracies, u32 num_epochs) {
    // lazy execution and only once, not really necessary as it is usually only called once, just future proofing
    static int term_width = -1;
    if (term_width == -1)  {
        struct winsize w;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0)
            term_width = w.ws_col;
    }

    // "Epoch XX: [" = 11, "]100.000000%" = 13
    int plot_size = term_width - 19;
    if (plot_size < 1) plot_size = 1;

    for (int i = 0; i < (int)num_epochs; i++) {
        printf("Epoch %d: [", i);
        int num_bars = (int)(accuracies[i] * plot_size);
        for (int j = 0; j < num_bars; j++) printf("=");
        for (int j = num_bars; j < plot_size; j++) printf(" ");
        printf("]%.2f%%\n", accuracies[i]*100.0);
    }
}

int main(int argc, char *argv[]) {
    // shitty but just assum that if the argumetn count is 5 the first two arguments are the training images and the second two arguments are the test images
    LabelledImages training_images;
    LabelledImages test_images;
    if (argc == 5) {
        if (!decodeImages(&training_images, argv[1], argv[2])) {
            printf("Error when reading the file");
            return 1;
        }
        if (!decodeImages(&test_images, argv[3], argv[4])) {
            printf("Error when reading the file");
            return 1;
        }
    } else {
        if (!decodeImages(&training_images, "./datasets/mnist/train-images.idx3-ubyte", "./datasets/mnist/train-labels.idx1-ubyte")) {
            printf("Error when reading the file");
            return 1;
        }
        if (!decodeImages(&test_images, "./datasets/mnist/t10k-images.idx3-ubyte", "./datasets/mnist/t10k-labels.idx1-ubyte")) {
            printf("Error when reading the file");
            return 1;
        }
    }
    
    // showImage(training_images.images);
    // hyper parameters
    // Epochs
    // learnign rate
    // layers
    // batchh size (1 rightfloat *accuracies now)
    // weight init
    // activations
    int epochs = 5;
    int batch_size = 64;
    float learning_rate = 0.001;
    
    Arena perm_arena = arenaCreate();
    Arena batch_arena = arenaCreate();
    
    NeuralNetwork network = {0};
    NeuralNetwork gradients = {0};

    // just a one layer network with output 10
    nnArenaCreate(&network, &perm_arena, 28*28, 2, 128, 10);
    
    // Xavier init: weights ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    for (int l = 0; l < network.num_layers; l++) {
        Layer *layer = &network.layers[l];
        float scale = 1.0f / sqrtf((float)layer->weights.cols);
        for (int i = 0; i < layer->weights.cols * layer->weights.rows; i++) {
            layer->weights.data[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
        }
    }
    
    // training
    Matrix in = matArenaCreate(&perm_arena, IMAGE_SIZE, 1);
    Matrix out = matArenaCreate(&perm_arena, 10, 1);
    Matrix actual = matArenaCreate(&perm_arena, 10, 1);
    matZero(&actual);
    float accuracies[epochs];
    
    u32 indeces[training_images.num_images];
    for (int i = 0; i < training_images.num_images; i++) {
        indeces[i] = i;
    }
    
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int epoch = 0; epoch < epochs; epoch++) {
        arenaReset(&batch_arena);
        
        shuffleIndeces(indeces, training_images.num_images);
        for (int i = 0; i < (int)training_images.num_images/batch_size; i++) {
            if (i % 100 == 0) {
                printf("Epoch %d, Batch %d\n", epoch, i);
            }

            assert(nnZeroGradients(&network, &gradients, &batch_arena) == 0);

            // the batch
            for (int j = 0; j < batch_size && i*batch_size + j < training_images.num_images; j++) {
                NeuralNetwork batch_gradients = {0};
                batch_gradients.num_layers = network.num_layers;

                Image *image = training_images.images + indeces[i*batch_size + j];
                u32 label = training_images.labels[indeces[i*batch_size + j]];

                // flattening the image
                for (int k = 0; k < IMAGE_SIZE; k++) {
                    in.data[k] = (float)image->data[k]/255.0;
                }
                
                assert(nnForward(&network, &out, in, &batch_arena) == 0);
                actual.data[label] = 1.0;
                assert(nnBackward(&network, &batch_gradients, actual, &batch_arena) == 0);
                actual.data[label] = 0.0;
                
                nnAddGradients(&gradients, &batch_gradients);
            }
            nnScaleGradients(&gradients, 1.0/batch_size);
            
            nnAddGradientsToNetwork(&network, &gradients, &batch_arena, learning_rate);
            arenaReset(&batch_arena);
        }
        //testing
        
        int total_correct = 0;
        for (int i = 0; i < (int)test_images.num_images; i++) {
            Image *image = test_images.images + i;
            u32 label = test_images.labels[i];
            
            // flattening the image
            for (int j = 0; j < IMAGE_SIZE; j++) {
                in.data[j] = (float)image->data[j]/255.0;
            }
            
            assert(nnForward(&network, &out, in, &batch_arena) == 0);
            
            u32 max_idx = 0;
            float curr_max = -1.0;
            for (int i = 0; i < 10; i++) {
                if (curr_max < out.data[i]) {
                    curr_max = out.data[i];
                    max_idx = i;
                }
            }
            total_correct += (max_idx == (int)label);
        }
        float accuracy = (float)total_correct / (float)test_images.num_images;
        accuracies[epoch] = accuracy;
        printf("Epoch %d, This model is %f%% accurate\n", epoch, accuracy*100.0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    printf("Training took %f seconds\n", (float)(end_time.tv_sec - start_time.tv_sec) + (float)(end_time.tv_nsec - start_time.tv_nsec)/1e9);
    
    plotAccuracies(accuracies, epochs);
    
    freeLabelledImages(training_images);
    freeLabelledImages(test_images);
    
    arenaDestroy(&batch_arena);
    arenaDestroy(&perm_arena);
    
    return 0;
}