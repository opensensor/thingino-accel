/*
 * NNA Descriptor Live Trace Tool
 * Monitors descriptor RAM while running OEM Venus inference
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dlfcn.h>

#define NNDMA_IO_PADDR       0x12508000
#define NNDMA_IO_SIZE        0x20
#define NNDMA_DESRAM_PADDR   0x12500000
#define NNDMA_DESRAM_SIZE    0x8000

/* Snapshot descriptor RAM and compare to find changes */
static uint8_t desram_snapshot[NNDMA_DESRAM_SIZE];
static int snapshot_valid = 0;

static void save_snapshot(void *desram) {
    memcpy(desram_snapshot, desram, NNDMA_DESRAM_SIZE);
    snapshot_valid = 1;
}

static void dump_changes(void *desram, const char *label) {
    if (!snapshot_valid) {
        save_snapshot(desram);
        return;
    }
    
    uint8_t *curr = (uint8_t *)desram;
    int changes = 0;
    
    printf("\n=== Descriptor RAM changes after %s ===\n", label);
    
    for (size_t i = 0; i < NNDMA_DESRAM_SIZE; i += 16) {
        int changed = 0;
        for (size_t j = 0; j < 16; j++) {
            if (curr[i + j] != desram_snapshot[i + j]) {
                changed = 1;
                break;
            }
        }
        if (changed) {
            printf("%04zx: ", i);
            for (size_t j = 0; j < 16; j++) {
                if (curr[i + j] != desram_snapshot[i + j])
                    printf("\033[1;31m%02x\033[0m ", curr[i + j]);
                else
                    printf("%02x ", curr[i + j]);
            }
            printf("\n");
            changes++;
        }
    }
    
    if (changes == 0) {
        printf("  (no changes)\n");
    } else {
        printf("  Total: %d lines changed\n", changes);
    }
    
    save_snapshot(desram);
}

static void dump_io_regs(volatile uint32_t *io) {
    printf("NNDMA I/O: ");
    for (int i = 0; i < 8; i++) {
        printf("[%x]=%08x ", i*4, io[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int memfd;
    void *io_map, *desram_map;
    
    printf("NNA Live Trace Tool\n");
    printf("====================\n\n");
    
    memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    
    io_map = mmap(NULL, NNDMA_IO_SIZE, PROT_READ | PROT_WRITE,
                  MAP_SHARED, memfd, NNDMA_IO_PADDR);
    desram_map = mmap(NULL, NNDMA_DESRAM_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, memfd, NNDMA_DESRAM_PADDR);
    
    if (io_map == MAP_FAILED || desram_map == MAP_FAILED) {
        perror("mmap");
        close(memfd);
        return 1;
    }
    
    printf("Mapped NNDMA I/O at %p, DESRAM at %p\n\n", io_map, desram_map);
    
    /* Take initial snapshot */
    printf("Taking initial snapshot...\n");
    dump_io_regs((volatile uint32_t *)io_map);
    save_snapshot(desram_map);
    
    /* Clear descriptor RAM to see what gets written */
    printf("Clearing DESRAM...\n");
    memset(desram_map, 0, NNDMA_DESRAM_SIZE);
    
    printf("\nWaiting for NNA operations (run Venus in another terminal)...\n");
    printf("Press Ctrl+C to exit\n\n");
    
    /* Poll for changes */
    while (1) {
        usleep(100000);  /* 100ms */
        
        volatile uint32_t *io = (volatile uint32_t *)io_map;
        static uint32_t last_io[8] = {0};
        
        int io_changed = 0;
        for (int i = 0; i < 8; i++) {
            if (io[i] != last_io[i]) {
                io_changed = 1;
                break;
            }
        }
        
        if (io_changed) {
            dump_io_regs(io);
            dump_changes(desram_map, "I/O change");
            memcpy(last_io, (void*)io, sizeof(last_io));
        }
    }
    
    munmap(desram_map, NNDMA_DESRAM_SIZE);
    munmap(io_map, NNDMA_IO_SIZE);
    close(memfd);
    
    return 0;
}

