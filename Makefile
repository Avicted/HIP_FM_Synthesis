# Compiler
HIPCC = hipcc

# MidiFile library path
MIDIFILE_LIB = submodules/midifile/lib/libmidifile.a

# Flags
HIPFLAGS = -O2 -ffp-contract=off

INCLUDES = -I/usr/include -I/opt/rocm/include -Isubmodules/midifile/include

LIBS = -Lsubmodules/midifile/lib -l:libmidifile.a

# Source files
SRC = src/Main.cu

# Executable
EXE = build/Main

# Build everything
all: midiFile $(EXE) 

# Build the main executable
$(EXE): $(SRC)
	$(shell rm -f *.wav)
	$(shell mkdir -p build)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES) -o $(EXE) $(SRC) $(LIBS)

# Build the MidiFile library
midiFile:
	@if [ ! -f $(MIDIFILE_LIB) ]; then \
		echo "Building MidiFile library..."; \
		cd submodules/midifile && make library -j8 && cd ../../; \
	else \
		echo "MidiFile library already built."; \
	fi

# Clean up
clean:
	rm -f $(EXE)
	rm -rf build
	$(MAKE) -C submodules/midifile clean

# Run the program
run: $(EXE)
	./$(EXE)

# Help message
help:
	@echo "Usage: make [all|clean|run|help]"
	@echo "  all:   Build the executable"
	@echo "  clean: Remove the executable"
	@echo "  run:   Run the executable"
	@echo "  help:  Display this help message"

.PHONY: all clean run help midiFile
