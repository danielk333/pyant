CC=gcc
CFLAGS=-fPIC
LIBS=-lm

SOURCES=$(wildcard *.c)
OBJECTS=$(SOURCES:.c=.o)
EXT=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTLIB=clibbeam$(EXT)

all: $(OUTLIB)

%.o: %.c
	@echo "pyant installation -> Compiling source file $< ..."
	$(CC) -c $(CFLAGS) -o $@ $<

$(OUTLIB): $(OBJECTS)
	@echo "pyant installation -> Linking shared library $@ ..."
	$(CC) $(CFLAGS) -shared $(OBJECTS) $(LIBS) -o $@
	@mv $@ ../pyant/ -v
	@echo "pyant installation -> The shared library $< has been created successfully."

clean:
	@echo "pyant installation -> Removing object files *.o ..."
	@-rm -f *.o
	@echo "pyant installation -> Removing shared library *.so ..."
	@-rm -f *.so
