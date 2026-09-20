CXX ?= c++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20 -march=native -Wall -Wextra

ARROW_PC := $(shell pkg-config --exists arrow && echo yes)
HOMEBREW_ARROW := /opt/homebrew/opt/apache-arrow

ifeq ($(ARROW_PC),yes)
  CPPFLAGS += $(shell pkg-config --cflags arrow)
  LDFLAGS += $(shell pkg-config --libs arrow)
else ifneq ($(wildcard $(HOMEBREW_ARROW)/include/arrow/api.h),)
  CPPFLAGS += -I$(HOMEBREW_ARROW)/include
  LDFLAGS += -L$(HOMEBREW_ARROW)/lib -Wl,-rpath,$(HOMEBREW_ARROW)/lib -larrow
else
  $(error Apache Arrow not found. On Ubuntu: sudo apt install -y g++ pkg-config libarrow-dev)
endif

BIN = mine_hard_negatives

.PHONY: all clean

all: $(BIN)

$(BIN): mine_hard_negatives.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BIN)
