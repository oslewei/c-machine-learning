
# Machine Learning in C

> **_NOTE:_** this is still a long work in progress

A neural network implementation in C used as a form to learn neural networks, gradients, and backpropagation. This was used as a learning project to teach myself machine learning, especially helpful as preliminary work for my Thesis which is heavily machine learning based.

## Evaluation

At current commit
- After all training images 1 epoch accurcy 86.81%
- After all training images 2 epochs accurcy 87.02%
- After 5 epochs it has decreased to 87.1 from 88, probably because it is not being shuffled

with shuffled indeces:

- Epoch 0: 90.06%, Epoch 1: 90.33%, Epoch 5: 88.91% 

With the Layers 28*28 -> 128 -> 10:

- Epoch 1: 75.84% accurate

With learning rate 0.01
After 1 epoch 97%

With the layers 28*28 -> 256 -> 128 -> 10

After 5 epochs 97.6%

With OpenBLAS and network 28*28 -> 128 -> 10 47.13 seconds 97.11% accuracy

### Fashion MNIST

Epoch 0: 72.439998%
Epoch 1: 81.209999%
Epoch 2: 81.569999%
Epoch 3: 81.779999%
Epoch 4: 82.330000%

## TODO

- [x] Backpropagation
- [x] Shuffling
- [x] Different sized models
- [x] Plotting accuracies
- [x] Batching (semi)
- [x] Automatic dataset downloading
- [x] Arena Allocation
- [x] OpenBLAS on matMul
- [x] Momentum
- [ ] Adam
- [ ] Convolutional neural networks
- [ ] Regularization
- [ ] Saving a model to disk (Either a straight dump or look into compatible filetypes, like gguf or pth)
- [ ] pthreads
- [ ] Different activation functions
- [ ] GPU acceleration

## Refactoring

- The output field in the matrix multiplication does not need to be a reference
- I dont like the bubbling up mechanic of errors anymore, combine with some logging mechanic
- I am not sure about the header only style

## Running

Only works on linux so far, does not have any external dependencies other than the c standard library

```bash
./datasets.sh mnist
make test
```
> this only works if you have the mnist dataset in a directory called mnist in the right format of idxn-ubyte. See the main.c file for the right file names.

## Resources

A great help to understand Backpropagation was [Neural Networks](http://neuralnetworksanddeeplearning.com/chap2.html)
