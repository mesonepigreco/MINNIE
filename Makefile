include make.inc
export
SUBDIRS = src


all : build

.PHONY: all build subdirs $(SUBDIRS) clean

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@


build: subdirs
	mkdir -p bin
	cp src/*.exe bin/

clean:
	rm -f bin/*.o bin/*.exe src/*.o src/*.exe

