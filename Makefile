# Makefile for building a CUDA program with hipify-clang (ROCm)

# Compilers
HIPCC = hipcc

# Flags
HIPFLAGS = -O2 -I/usr/include -I/opt/rocm/include

# Source files
SRC = src/Main.cu

# Executable
EXE = build/Main

# Build
all: $(EXE) run

$(EXE): $(SRC)
	# Create build directory
	$(shell mkdir -p build)
	$(HIPCC) $(HIPFLAGS) -o $(EXE) $(SRC)

# Clean
clean:
	rm -f $(EXE)

# Run
run: $(EXE)
	./$(EXE)

# Help
help:
	@echo "Usage: make [all|clean|run|help]"
	@echo "  all:   Build the executable"
	@echo "  clean: Remove the executable"
	@echo "  run:   Run the executable"
	@echo "  help:  Display this help message"

.PHONY: all clean run help
