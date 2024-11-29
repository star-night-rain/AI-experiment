CXX = g++
CXXFLAGS = -std=c++17 -O3 

# Source files and target files
SRC = Graph.cpp main.cpp
TARGET = run

# Generate target
all: $(TARGET)

# Compilation rule
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Clean rule
clean:
	@echo "Cleaning up..."
	-@rm -f *.o *.gcno *~ SOR shortestPath SRH Statistic 2>/dev/null || true

.PHONY: all clean
