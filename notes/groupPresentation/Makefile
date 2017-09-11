
all: groupPresentation

groupPresentation: groupPresentation.md
	pandoc -t beamer --template=beamerModified -H ~/.pandoc/preamble.tex -o $@.pdf $@.md --verbose
