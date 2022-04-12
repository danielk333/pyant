CC=gcc
CFLAGS=
LIBS=-lm
SOURCES:=$(wildcard *.c)

build: $(SOURCES)
	gcc $(LIBS) -o libbeam.so $(SOURCES) $(CFLAGS)

.PHONY: clean

clean:
	rm -v *.so