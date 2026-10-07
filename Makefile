CFLAGS = -pedantic -Wall -g
# All the SRCS with implementationn
SRC_FILES = main.c arena.h neuralnetwork.h matrix.h
SRCS = $(patsubst %,src/%,$(SRC_FILES))

run: build-main
	./main

test: build-test
	./test

build-main: $(SRCS)
	gcc $(CFLAGS) -fsanitize=address -o main src/main.c -lm

build-test: $(SRCS)
	gcc $(CFLAGS) -O3 -ffast-math -march=native -o test src/main.c -lm

build-cblas: $(SRCS)
	gcc $(CFLAGS) -O3 -ffast-math -march=native -DUSE_OPENBLAS -o cblas src/main.c -lm -lopenblas

leak: $(SRCS)
	gcc -pedantic -Wall -g -o leak src/main.c -lm
	valgrind --leak-check=full ./leak
	rm leak

.PHONY: clean leak build-test build-main main test
clean:
	rm -f main test leak
