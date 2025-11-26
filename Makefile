# thingino-accel Makefile
# Build system for musl-based cross-compilation

# Toolchain configuration
CROSS_COMPILE ?= mipsel-linux-
CC := $(CROSS_COMPILE)gcc
AR := $(CROSS_COMPILE)ar
STRIP := $(CROSS_COMPILE)strip

# Directories
SRC_DIR := src
INC_DIR := include
EXAMPLES_DIR := examples
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin

# Compiler flags
CFLAGS := -Wall -Wextra -O2 -fPIC
CFLAGS += -I$(INC_DIR)
LDFLAGS := -L$(LIB_DIR)
LIBS := -lpthread

# Library name
LIB_NAME := libnna
LIB_STATIC := $(LIB_DIR)/$(LIB_NAME).a
LIB_SHARED := $(LIB_DIR)/$(LIB_NAME).so

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# Example programs
EXAMPLES := test_init
EXAMPLE_BINS := $(patsubst %,$(BIN_DIR)/%,$(EXAMPLES))

# Targets
.PHONY: all clean lib examples install

all: lib examples

# Create directories
$(OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build static library
$(LIB_STATIC): $(OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^
	@echo "Built static library: $@"

# Build shared library
$(LIB_SHARED): $(OBJS) | $(LIB_DIR)
	$(CC) -shared -o $@ $^ $(LIBS)
	@echo "Built shared library: $@"

# Build library (both static and shared)
lib: $(LIB_STATIC) $(LIB_SHARED)

# Build examples
$(BIN_DIR)/%: $(EXAMPLES_DIR)/%.c $(LIB_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) -lnna $(LIBS)
	@echo "Built example: $@"

examples: $(EXAMPLE_BINS)

# Install (for cross-compilation, just copy to build dir)
install: all
	@echo "Library and examples built in $(BUILD_DIR)"
	@echo "To deploy to device:"
	@echo "  scp $(LIB_SHARED) root@device:/usr/lib/"
	@echo "  scp $(BIN_DIR)/test_init root@device:/tmp/"

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Show configuration
info:
	@echo "Toolchain: $(CROSS_COMPILE)"
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "Sources: $(SRCS)"
	@echo "Objects: $(OBJS)"
	@echo "Examples: $(EXAMPLES)"

# Dependencies
-include $(OBJS:.o=.d)

# Generate dependencies
$(OBJ_DIR)/%.d: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@$(CC) $(CFLAGS) -MM -MT $(OBJ_DIR)/$*.o $< > $@

