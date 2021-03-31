
all: bin/jpegff bin/ffjpeg
.PHONY: all

bin/jpegff: jpegff.c
	gcc -o $@ $^ -lm

bin/ffjpeg: ffjpeg.c
	gcc -o $@ $^ -lm
