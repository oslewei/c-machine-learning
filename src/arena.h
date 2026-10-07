#ifndef ARENA_H
#define ARENA_H

#include <assert.h>
#include <stddef.h> 
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// 16KB initial memory
#define ARENA_DEFAULT_SIZE (16*1024)

typedef struct ArenaNode {
    struct ArenaNode *next;
    size_t len;
    size_t cap;
    char data[];
}  ArenaNode;

typedef struct {
    ArenaNode *start;
    ArenaNode *end;
} Arena;

// returns NULL on error
static ArenaNode *arenaNodeCreate(size_t size) {
    ArenaNode *node = (ArenaNode *)malloc(sizeof(ArenaNode)+size);
    if (node == NULL) return NULL;
    node->cap = size;
    node->len = 0;
    node->next = NULL;
    
    return node;
}

static void arenaNodeDestroy(ArenaNode *node) {
    free(node);
}

// 0 initialized
static Arena arenaCreate() {
    ArenaNode *node = arenaNodeCreate(ARENA_DEFAULT_SIZE);
    
    return (Arena){
        .start = node,
        .end = node,
    };
}

// Not zero initialized
static void *arenaAlloc(Arena *arena, size_t size) {
    size_t align = alignof(max_align_t);
    
    uintptr_t current = (uintptr_t)(arena->end->data + arena->end->len);
    uintptr_t aligned = (current + align - 1) & ~(align - 1);
    
    size_t padding = aligned - current;
    
    if (arena->end->len + padding + size > arena->end->cap) {
        // allocate more
        // Expands the size to the max of the size or default size
        size_t alloc_size = ARENA_DEFAULT_SIZE;
        if (alloc_size < size + align) alloc_size = size + align;
        ArenaNode *node = arenaNodeCreate(alloc_size);
        if (node == NULL) return NULL;
        arena->end->next = node;
        arena->end = node;

        // Recalculate alignment for the new node
        current = (uintptr_t)(arena->end->data + arena->end->len);
        aligned = (current + align - 1) & ~(align - 1);
        padding = aligned - current;
    }

    arena->end->len += padding + size;
    return (void *)(aligned);
}

static void arenaDestroy(Arena *arena) {
    ArenaNode *node = arena->start;
    while (node != NULL) {
        ArenaNode *next = node->next;
        arenaNodeDestroy(node);
        node = next;
    }
    arena->start = NULL;
    arena->end = NULL;
}

static void arenaReset(Arena *arena) {
    // Free all nodes beyond the first
    ArenaNode *node = arena->start->next;
    while (node != NULL) {
        ArenaNode *next = node->next;
        arenaNodeDestroy(node);
        node = next;
    }
    arena->start->next = NULL;
    arena->start->len = 0;
    arena->end = arena->start;
}


#endif