TEX := thesis_progress

.PHONY: all clean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX).tex

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc
