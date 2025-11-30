/*
 * NNA Descriptor Trace Tool
 * Dumps the NNDMA descriptor RAM and I/O registers to understand the format
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#define NNDMA_IO_PADDR       0x12508000  /* NNA DMA I/O registers */
#define NNDMA_IO_SIZE        0x20        /* 32 bytes */
#define NNDMA_DESRAM_PADDR   0x12500000  /* NNA DMA descriptor RAM */
#define NNDMA_DESRAM_SIZE    0x8000      /* 32KB */

static void hexdump(const char *name, void *addr, size_t len) {
    uint8_t *p = (uint8_t *)addr;
    printf("\n=== %s (0x%zx bytes) ===\n", name, len);
    for (size_t i = 0; i < len; i += 16) {
        printf("%04zx: ", i);
        for (size_t j = 0; j < 16 && i + j < len; j++) {
            printf("%02x ", p[i + j]);
        }
        printf(" ");
        for (size_t j = 0; j < 16 && i + j < len; j++) {
            char c = p[i + j];
            printf("%c", (c >= 32 && c < 127) ? c : '.');
        }
        printf("\n");
    }
}

static void dump_nndma_io(volatile uint32_t *io) {
    printf("\n=== NNDMA I/O Registers ===\n");
    for (int i = 0; i < 8; i++) {
        printf("  [0x%02x] = 0x%08x\n", i * 4, io[i]);
    }
}

int main(int argc, char **argv) {
    int memfd;
    void *io_map, *desram_map;
    
    printf("NNA Descriptor Trace Tool\n");
    printf("==========================\n\n");
    
    /* Open /dev/mem */
    memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) {
        perror("Failed to open /dev/mem");
        return 1;
    }
    
    /* Map NNDMA I/O registers */
    io_map = mmap(NULL, NNDMA_IO_SIZE, PROT_READ | PROT_WRITE,
                  MAP_SHARED, memfd, NNDMA_IO_PADDR);
    if (io_map == MAP_FAILED) {
        perror("Failed to map NNDMA I/O");
        close(memfd);
        return 1;
    }
    printf("NNDMA I/O mapped at %p (phys 0x%08x)\n", io_map, NNDMA_IO_PADDR);
    
    /* Map NNDMA descriptor RAM */
    desram_map = mmap(NULL, NNDMA_DESRAM_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, memfd, NNDMA_DESRAM_PADDR);
    if (desram_map == MAP_FAILED) {
        perror("Failed to map NNDMA DESRAM");
        munmap(io_map, NNDMA_IO_SIZE);
        close(memfd);
        return 1;
    }
    printf("NNDMA DESRAM mapped at %p (phys 0x%08x)\n", desram_map, NNDMA_DESRAM_PADDR);
    
    /* Dump I/O registers */
    dump_nndma_io((volatile uint32_t *)io_map);
    
    /* Dump first 512 bytes of descriptor RAM */
    hexdump("NNDMA DESRAM (first 512 bytes)", desram_map, 512);
    
    /* Look for non-zero regions in descriptor RAM */
    printf("\n=== Scanning DESRAM for non-zero regions ===\n");
    uint32_t *p = (uint32_t *)desram_map;
    int regions_found = 0;
    for (size_t i = 0; i < NNDMA_DESRAM_SIZE / 4; i++) {
        if (p[i] != 0) {
            size_t start = i;
            while (i < NNDMA_DESRAM_SIZE / 4 && p[i] != 0) i++;
            size_t end = i;
            printf("  Non-zero region: offset 0x%04zx - 0x%04zx (%zu words)\n",
                   start * 4, end * 4, end - start);
            if (regions_found < 3) {
                hexdump("Region content", &p[start], (end - start) * 4 > 256 ? 256 : (end - start) * 4);
            }
            regions_found++;
        }
    }
    if (regions_found == 0) {
        printf("  (all zeros - NNA not currently in use)\n");
    }
    
    /* Cleanup */
    munmap(desram_map, NNDMA_DESRAM_SIZE);
    munmap(io_map, NNDMA_IO_SIZE);
    close(memfd);
    
    return 0;
}

