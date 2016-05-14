# C++ source code
CXXSRC	= NUC.cpp \
	borders.cpp \
		MIRE.cpp
# all source code
SRC	=  $(CXXSRC)

# C++ objects
CXXOBJ	= $(CXXSRC:.cpp=.o)

# all objects
OBJ	= $(CXXOBJ)

# binary target
BIN	= demo_NUC

default	: $(BIN)

# C++ optimization flags
CXXOPT	= -O3 -ftree-vectorize -funroll-loops

# C++ compilation flags
CXXFLAGS	= $(CXXOPT) -Wall -Wextra \
	-Wno-write-strings -Wno-deprecated -ansi `pkg-config --libs opencv`

# link flags
LDFLAGS	= -lm



# use openMP with `make OMP=1`
ifdef OMP
CXXFLAGS	+= -fopenmp
LDFLAGS += -lgomp
else
CXXFLAGS  += -Wno-unknown-pragmas
endif


# use DEBUG
ifdef DEBUG
CXXFLAGS	+= -g
LDFLAGS += -g
endif


# partial compilation of C++ source code
%.o: %.cpp %.h
	$(CXX) -c -o $@  $< 

# link all the opject code
$(BIN): $(OBJ) $(LIBDEPS)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS) $(CXXFLAGS)

# housekeeping
.PHONY	: clean distclean
clean	:
	$(RM) $(OBJ)
	
distclean	: clean
	$(RM) $(BIN)
	
