
all: src
	mkdir -p bin
	cp src/*.exe bin/

src:
	cd src
	$(MAKE) -C $@

