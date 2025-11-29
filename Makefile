# thingino-accel Makefile
# Build system for musl-based cross-compilation

# Toolchain configuration
CROSS_COMPILE ?= mipsel-linux-
CC := $(CROSS_COMPILE)gcc
CXX := $(CROSS_COMPILE)g++
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
CFLAGS += -I$(INC_DIR) -I$(SRC_DIR)
CXXFLAGS := -Wall -Wextra -O2 -fPIC -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=1
CXXFLAGS += -I$(INC_DIR) -I$(SRC_DIR)
LDFLAGS := -L$(LIB_DIR)
LIBS := -lpthread -lstdc++ -ldl

# Library names
LIB_NNA_NAME := libnna
LIB_NNA_STATIC := $(LIB_DIR)/$(LIB_NNA_NAME).a
LIB_NNA_SHARED := $(LIB_DIR)/$(LIB_NNA_NAME).so

LIB_VENUS_NAME := libvenus
LIB_VENUS_STATIC := $(LIB_DIR)/$(LIB_VENUS_NAME).a
LIB_VENUS_SHARED := $(LIB_DIR)/$(LIB_VENUS_NAME).so

# Source files
C_SRCS := $(wildcard $(SRC_DIR)/*.c)
C_OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(C_SRCS))

CXX_SRCS := $(wildcard $(SRC_DIR)/venus/*.cpp)
CXX_OBJS := $(patsubst $(SRC_DIR)/venus/%.cpp,$(OBJ_DIR)/venus_%.o,$(CXX_SRCS))

# Mars sources
MARS_SRCS := $(wildcard $(SRC_DIR)/mars/*.c)
MARS_OBJS := $(patsubst $(SRC_DIR)/mars/%.c,$(OBJ_DIR)/mars_%.o,$(MARS_SRCS))

ALL_OBJS := $(C_OBJS) $(CXX_OBJS)

# Example programs (C)
EXAMPLES := test_init test_model_load test_inference
EXAMPLE_BINS := $(patsubst %,$(BIN_DIR)/%,$(EXAMPLES))

# Example programs (C++)
CXX_EXAMPLES := yolo_detect
CXX_EXAMPLE_BINS := $(patsubst %,$(BIN_DIR)/%,$(CXX_EXAMPLES))

# Targets
.PHONY: all clean lib examples install

all: lib examples

# Create directories
$(OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile C object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile C++ object files (Venus)
$(OBJ_DIR)/venus_%.o: $(SRC_DIR)/venus/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile Mars C files
$(OBJ_DIR)/mars_%.o: $(SRC_DIR)/mars/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build NNA static library (includes C++ Venus objects)
$(LIB_NNA_STATIC): $(C_OBJS) $(CXX_OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^
	@echo "Built static library: $@"

# Build NNA shared library (includes C++ Venus objects)
$(LIB_NNA_SHARED): $(C_OBJS) $(CXX_OBJS) | $(LIB_DIR)
	$(CXX) -shared -o $@ $^ $(LIBS)
	@echo "Built shared library: $@"

# Build Venus static library
$(LIB_VENUS_STATIC): $(CXX_OBJS) $(C_OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^
	@echo "Built static library: $@"

# Build Venus shared library
$(LIB_VENUS_SHARED): $(CXX_OBJS) $(C_OBJS) | $(LIB_DIR)
	$(CXX) -shared -o $@ $^ $(LIBS)
	@echo "Built shared library: $@"

# Build libraries (both NNA and Venus)
lib: $(LIB_NNA_STATIC) $(LIB_NNA_SHARED) $(LIB_VENUS_STATIC) $(LIB_VENUS_SHARED)

# Build C examples
$(BIN_DIR)/%: $(EXAMPLES_DIR)/%.c $(LIB_NNA_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) -lnna $(LIBS)
	@echo "Built example: $@"

# Build C++ examples
$(BIN_DIR)/yolo_detect: $(EXAMPLES_DIR)/yolo_detect.cpp $(LIB_NNA_STATIC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) -lnna $(LIBS) -lm
	@echo "Built C++ example: $@"

# Build Mars test (filter out mars_test from MARS_OBJS for library)
MARS_LIB_OBJS := $(filter-out $(OBJ_DIR)/mars_mars_test.o,$(MARS_OBJS))

$(BIN_DIR)/mars_test: $(SRC_DIR)/mars/mars_test.c $(OBJ_DIR)/mars_mars_runtime.o $(LIB_NNA_STATIC) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(OBJ_DIR)/mars_mars_runtime.o -o $@ $(LDFLAGS) -lnna $(LIBS) -lm
	@echo "Built Mars test: $@"

examples: $(EXAMPLE_BINS) $(CXX_EXAMPLE_BINS) $(BIN_DIR)/mars_test

# Install (for cross-compilation, just copy to build dir)
install: all
	@echo "Libraries and examples built in $(BUILD_DIR)"
	@echo "To deploy to device:"
	@echo "  scp $(LIB_NNA_SHARED) $(LIB_VENUS_SHARED) root@device:/usr/lib/"
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

