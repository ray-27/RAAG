CXX ?= clang++
ARROW_PREFIX ?= /opt/homebrew/opt/apache-arrow
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -march=native -Wall -Wextra
CPPFLAGS += -I$(ARROW_PREFIX)/include -I/opt/homebrew/include
LDFLAGS += -L$(ARROW_PREFIX)/lib -Wl,-rpath,$(ARROW_PREFIX)/lib -larrow

BIN = mine_hard_negatives

.PHONY: all clean

all: $(BIN)

$(BIN): mine_hard_negatives.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BIN)
