## Makefile for neural network controller simulation

CC        = g++
CXXFLAGS 		= -Wall -Wextra -std=c++11 -O3 -DNDEBUG -DSCOTS_BDD
#
# scots 
#
SCOTSROOT		= ../../ext/SCOTSv0.2
SCOTSINC		= -I$(SCOTSROOT)/src -I$(SCOTSROOT)/utils
#
# cudd 
#
CUDD		    =  -I/usr/local/include -L/usr/local/lib -lcudd

.PHONY: simulation
	
TARGET = simulation

all: $(TARGET)

simulation: 
	$(CC) simulation.cc -o simulation $(CXXFLAGS) $(SCOTSINC) $(CUDD) 

clean:
	rm simulation
