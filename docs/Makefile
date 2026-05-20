TEX  = reconexion_magnetica
DOCS = .

.PHONY: all clean

all:
	pdflatex $(TEX).tex
	pdflatex $(TEX).tex   # segunda pasada para TOC y refs cruzadas

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).toc $(TEX).out
