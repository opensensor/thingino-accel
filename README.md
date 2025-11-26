# thingino-accel

Open-source Neural Network Accelerator (NNA) library for Ingenic T41/T31 platforms running Thingino firmware.

## Overview

This is a clean-room implementation of the Ingenic Magik Venus NNA inference library, designed to work with musl libc instead of requiring glibc/uclibc. It provides hardware-accelerated neural network inference on Ingenic SoCs.

## Status

🚧 **Early Development** - Basic functionality being implemented

### Current Features
- [ ] NNA device initialization (`/dev/soc-nna`)
- [ ] ORAM memory management
- [ ] DDR memory allocation
- [ ] Basic tensor operations
- [ ] Model loading (.mgk format)
- [ ] Inference execution

## Architecture

```
thingino-accel/
├── include/           # Public API headers
│   ├── nna.h         # Main NNA interface
│   ├── tensor.h      # Tensor operations
│   └── memory.h      # Memory management
├── src/              # Implementation
│   ├── device.c      # /dev/soc-nna interface
│   ├── memory.c      # ORAM/DDR allocation
│   ├── tensor.c      # Tensor implementation
│   └── model.c       # Model loading
├── examples/         # Example programs
│   └── test_init.c   # Basic initialization test
└── build/            # Build output

```

## Building

```bash
# Set up cross-compilation environment
export CROSS_COMPILE=mipsel-linux-
export CC=${CROSS_COMPILE}gcc
export AR=${CROSS_COMPILE}ar

# Build library
make

# Build examples
make examples
```

## Hardware Support

- **Ingenic T41** (XBurst2) - Primary target
- **Ingenic T31** - Future support

### Memory Layout (T41)
- ORAM: 384 KB @ 0x12620000 (on-chip accelerator RAM)
- DMA Descriptors: 16 KB @ 0x12500000

## Usage Example

```c
#include <nna.h>

int main() {
    // Initialize NNA hardware
    if (nna_init() != 0) {
        fprintf(stderr, "Failed to initialize NNA\n");
        return 1;
    }
    
    // Allocate memory for tensors
    void *input_data = nna_malloc(1024);
    
    // ... perform inference ...
    
    // Cleanup
    nna_free(input_data);
    nna_deinit();
    
    return 0;
}
```

## Differences from OEM libvenus

| Feature | OEM libvenus | thingino-accel |
|---------|--------------|----------------|
| C Library | glibc/uclibc | musl |
| Language | C++ | C |
| Size | ~400 KB | Target: <100 KB |
| API | C++ classes | C API |
| License | Proprietary | Open Source |

## Development Approach

This library is being developed through:
1. **Reverse engineering** OEM binaries (libvenus.so, libdrivers.so)
2. **Reference documentation** from Ingenic Magik toolkit
3. **Hardware testing** on Wyze Cam V4 (T41 platform)
4. **Iterative development** - start minimal, expand functionality

## Testing

Run on device:
```bash
# Load kernel module
insmod /lib/modules/soc-nna.ko

# Run test
./test_init
```

## Contributing

This is an early-stage project. Contributions welcome!

## References

- Ingenic Magik Venus Programming Manual
- OEM binaries: `libvenus.so`, `libdrivers.so`, `libaip.so`
- Kernel module: `soc-nna.ko`
- Model format: `.mgk` (ELF shared object)

## License

TBD - Open source license to be determined

