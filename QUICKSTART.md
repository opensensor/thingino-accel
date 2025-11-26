# Quick Start Guide

## Prerequisites

- Thingino firmware on Ingenic T41/T31 device
- Cross-compilation toolchain (mipsel-linux-musl)
- SSH access to device

## Build

```bash
cd thingino-accel
./build.sh
```

This will create:
- `build/lib/libnna.so` - Shared library
- `build/lib/libnna.a` - Static library
- `build/bin/test_init` - Test program

## Deploy to Device

```bash
# Replace with your device IP
DEVICE_IP=192.168.x.x

# Copy library and test
scp build/lib/libnna.so root@$DEVICE_IP:/usr/lib/
scp build/bin/test_init root@$DEVICE_IP:/tmp/
```

## Run Test

```bash
ssh root@$DEVICE_IP

# Load NNA kernel module (if not already loaded)
insmod /lib/modules/soc-nna.ko

# Run test
/tmp/test_init
```

Expected output:
```
╔══════════════════════════════════════════════════════════╗
║         thingino-accel - Initialization Test            ║
╚══════════════════════════════════════════════════════════╝

[1/6] Library version
      Version: 0.1.0-dev
  ✓ Version retrieved

[2/6] NNA initialization
NNA initialized: ORAM @ 0x12620000 (384 KB)
  ✓ NNA initialized

[3/6] Hardware information
      ORAM Physical: 0x12620000
      ORAM Virtual:  0xb6f00000
      ORAM Size:     384 KB
      NNA Version:   0x20
  ✓ Hardware info retrieved

[4/6] DDR memory allocation
      Allocated: 1048576 bytes @ 0x77f00000
  ✓ DDR memory is writable
  ✓ DDR memory freed

[5/6] ORAM allocation
      Allocated: 4096 bytes @ 0xb6f00000
      ORAM usage: 4 / 384 KB (1.0%)
  ✓ ORAM allocation successful

[6/6] Tensor operations
      Shape: [1, 224, 224, 3]
      Elements: 150528
      Bytes: 150528
  ✓ Tensor created
  ✓ Tensor data written
  ✓ Tensor destroyed

[CLEANUP] Shutting down
  ✓ NNA deinitialized

╔══════════════════════════════════════════════════════════╗
║              ✓ ALL TESTS PASSED!                        ║
╚══════════════════════════════════════════════════════════╝
```

## Using the Library

### Basic Example

```c
#include <nna.h>
#include <nna_memory.h>
#include <nna_tensor.h>

int main() {
    // Initialize NNA
    if (nna_init() != NNA_SUCCESS) {
        return 1;
    }
    
    // Create input tensor (1x224x224x3 RGB image)
    nna_shape_t shape = nna_shape_make(1, 224, 224, 3);
    nna_tensor_t *input = nna_tensor_create(
        &shape, 
        NNA_DTYPE_UINT8, 
        NNA_FORMAT_NHWC
    );
    
    // Fill with image data
    unsigned char *data = nna_tensor_data(input);
    // ... load image into data ...
    
    // TODO: Load model and run inference
    
    // Cleanup
    nna_tensor_destroy(input);
    nna_deinit();
    
    return 0;
}
```

### Compile Your Program

```bash
mipsel-linux-gcc -o myapp myapp.c \
    -I/path/to/thingino-accel/include \
    -L/path/to/thingino-accel/build/lib \
    -lnna -lpthread
```

## Troubleshooting

### "Failed to open /dev/soc-nna"

Make sure the kernel module is loaded:
```bash
lsmod | grep soc-nna
# If not loaded:
insmod /lib/modules/soc-nna.ko
```

### "Out of ORAM memory"

ORAM is limited (384 KB on T41). Use DDR for large allocations:
```c
void *large_buffer = nna_malloc(1024 * 1024);  // DDR
void *small_buffer = nna_oram_malloc(4096);    // ORAM
```

### Build Errors

Make sure toolchain is in PATH:
```bash
export PATH="/path/to/thingino/toolchain/bin:$PATH"
export CROSS_COMPILE=mipsel-linux-
```

## Next Steps

1. **Explore the API** - See `include/` directory
2. **Read DEVELOPMENT.md** - Learn about internals
3. **Check REVERSE_ENGINEERING.md** - Understand OEM library
4. **Contribute** - Help implement model loading and inference!

## API Reference

### Core Functions (nna.h)
- `nna_init()` - Initialize NNA
- `nna_deinit()` - Cleanup
- `nna_get_hw_info()` - Get hardware info
- `nna_is_ready()` - Check if initialized

### Memory (nna_memory.h)
- `nna_malloc()` - Allocate DDR memory
- `nna_free()` - Free memory
- `nna_oram_malloc()` - Allocate ORAM
- `nna_oram_free()` - Free ORAM

### Tensors (nna_tensor.h)
- `nna_tensor_create()` - Create tensor
- `nna_tensor_destroy()` - Destroy tensor
- `nna_tensor_data()` - Get data pointer
- `nna_shape_make()` - Create shape

## Support

- GitHub Issues: (TBD)
- Documentation: See `*.md` files in repo
- Examples: See `examples/` directory

