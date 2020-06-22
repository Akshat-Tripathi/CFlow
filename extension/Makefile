CC      = gcc
CFLAGS  = -Wall -g -D_DEFAULT_SOURCE -pedantic -std=c99
LDLIBS=-lm

.SUFFIXES: .c .o

.PHONY: all clean

all: c/demo c/test

c/demo: c/nodes.o c/testUtils.o c/util.o c/graphix.o c/scheduler.o c/activation.o c/matrix.o c/layers.o c/predict.o c/error.o c/compiler.o c/file.o c/data.o c/optimisers.o c/train.o c/readCSV.o

c/train: c/train.o c/nodes.o c/testUtils.o c/util.o c/graphix.o c/scheduler.o c/activation.o c/matrix.o c/layers.o c/predict.o c/error.o c/compiler.o c/optimisers.o

c/test: c/test.o c/nodes.o c/activation.o c/testUtils.o c/file.o c/scheduler.o c/matrix.o c/util.o c/data.o c/error.c c/optimisers.c c/readCSV.o

c/test.o: nodes.h activation.h testUtils.h file.h scheduler.h matrix.h data.h error.h optimisers.h readCSV.h

c/activation.o: activation.h

c/data.o: data.h nodes.h matrix.h

c/file.o: file.h nodes.h data.h matrix.h util.h testUtils.h

c/graphix.o: graphix.h nodes.h util.h

c/matrix.o: matrix.h activation.h

c/nodes.o: nodes.h matrix.h

c/optimisers.o:

c/predict.o: predict.h data.h matrix.h nodes.h util.h

c/readCSV.o: readCSV.h matrix.h

c/scheduler.o: scheduler.h nodes.h util.h testUtils.h

c/testUtils.o: testUtils.h nodes.h scheduler.h util.h

c/util.o: util.h nodes.h

c/graphix.o: graphix.h nodes.h

clean:
	rm -f $(wildcard *.o)
	rm -f $(wildcard */*.o)
	rm -f $(wildcard */*/*.o)
	rm -f c/test
	rm -f c/demo
