# Compiler
NVCC=nvcc

# Compiler flags
# -O3 for optimization we use A100 so sm_80
NVCC_FLAGS=-O3 -arch=sm_80

# Target executable name
TARGET=wa1-task3

# Default rule
all: $(TARGET)

# Rule to build the executable from the .cu file
$(TARGET): wa1-task3.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Rule to clean up the build directory
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean

