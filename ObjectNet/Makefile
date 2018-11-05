CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g
LDFLAGS=-g
LDLIBS=

SRCS=mda_mnist.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: mda_mnist

mda_mnist: $(OBJS)
	$(CXX) $(LDFLAGS) -o mda_mnist $(OBJS) $(LDLIBS) 

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) *~ .depend

include .depend