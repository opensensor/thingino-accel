#!/bin/bash
#
# Quick build script for thingino-accel
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         thingino-accel - Build Script                   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect toolchain
if [ -z "$CROSS_COMPILE" ]; then
    # Try to find thingino toolchain
    if [ -d "/home/matteius/output-stable/wyze_camv4_t41nq_gc4653_atbm6062/host/bin" ]; then
        export PATH="/home/matteius/output-stable/wyze_camv4_t41nq_gc4653_atbm6062/host/bin:$PATH"
        export CROSS_COMPILE=mipsel-linux-
        echo -e "${YELLOW}Using thingino toolchain${NC}"
    else
        echo -e "${RED}Error: CROSS_COMPILE not set and thingino toolchain not found${NC}"
        echo "Please set CROSS_COMPILE environment variable"
        exit 1
    fi
fi

# Show configuration
echo "Toolchain: ${CROSS_COMPILE}"
echo "CC: ${CROSS_COMPILE}gcc"
echo ""

# Build
echo -e "${GREEN}[1/3] Building library...${NC}"
make lib

echo ""
echo -e "${GREEN}[2/3] Building examples...${NC}"
make examples

echo ""
echo -e "${GREEN}[3/3] Build complete!${NC}"
echo ""

# Show results
echo "Built files:"
ls -lh build/lib/
ls -lh build/bin/
echo ""

# Show deployment instructions
echo -e "${YELLOW}To deploy to device:${NC}"
echo "  scp build/lib/libnna.so root@<device-ip>:/usr/lib/"
echo "  scp build/bin/test_init root@<device-ip>:/tmp/"
echo ""
echo "To run on device:"
echo "  ssh root@<device-ip>"
echo "  insmod /lib/modules/soc-nna.ko  # if not already loaded"
echo "  /tmp/test_init"
echo ""

