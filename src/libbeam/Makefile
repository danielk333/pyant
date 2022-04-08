CC=gcc
CFLAGS=
LIBS=-lm

build: libbeam.c
	gcc $(LIBS) -o libbeam.so libbeam.c $(CFLAGS)

.PHONY: clean

clean:
	rm -v *.so