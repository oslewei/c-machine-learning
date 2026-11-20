CFLAGS = -pedantic -Wall -g
LIBS = -lm
BUILD = build

SRC_FILES = main.c arena.h neuralnetwork.h matrix.h
SRCS = $(addprefix src/,$(SRC_FILES))

BENCH_FILES = benchmarks/bench.c benchmarks/bench.h

TEST_FILES = tests/test.h tests/test_runner.c tests/test_matrix.c

all: build-main

run: build-main
	./$(BUILD)/main

build-main: $(SRCS) | $(BUILD)
	gcc $(CFLAGS) -fsanitize=address -o $(BUILD)/main src/main.c $(LIBS)

train: build-train
	./$(BUILD)/train

build-train: $(SRCS) | $(BUILD)
	gcc $(CFLAGS) -O3 -ffast-math -march=native -o $(BUILD)/train src/main.c $(LIBS)

build-cblas: $(SRCS) | $(BUILD)
	gcc $(CFLAGS) -O3 -ffast-math -march=native -DUSE_OPENBLAS -o $(BUILD)/cblas src/main.c $(LIBS) -lopenblas

test: build-test
	./$(BUILD)/test

build-test: $(TEST_FILES) | $(BUILD)
	gcc $(CFLAGS) -o $(BUILD)/test tests/test_runner.c tests/test_matrix.c $(LIBS)

build-bench: $(SRCS) $(BENCH_FILES) | $(BUILD)
	gcc $(CFLAGS) -O3 -march=native -o $(BUILD)/bench benchmarks/bench.c $(LIBS)

leak: $(SRCS) | $(BUILD)
	gcc $(CFLAGS) -o $(BUILD)/leak src/main.c $(LIBS)
	valgrind --leak-check=full ./$(BUILD)/leak

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all run train test leak clean
